#coding=utf-8
#__author__='YHR'
from second_Stage_Params import Params
from second_Stage_classDefine import dn as mymodel
from helpFunc import create_path
from getData import GetBatchTask, GetTestData_precModify, GetYanzhengData_precModify

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report

def testIntrainProcess(mm, sess, tsttrainX, tsttrainY, tstbatchNum):
    train_All_Loss = []
    for _B in range(tstbatchNum):
        train_x = tsttrainX[_B]
        train_label = tsttrainY[_B]

        feedDict = {mm.train_x: train_x,
                    mm.train_label: train_label,
                    mm.Batchtrain: False, mm.dropouttrain: False}

        trainLoss = sess.run([mm.only_train_cls_trainloss], feed_dict=feedDict)
        train_All_Loss.append(trainLoss)
    mean_train_All_Loss= np.mean(train_All_Loss)
    return mean_train_All_Loss


def choostEmphasizeSet(sess, mm, posX, negX, params):
    pos_aimNum = params.batch_samples // 2
    neg_aimNum = params.batch_samples - pos_aimNum
    # pos
    feedDict = {mm.tstx: posX,  mm.Batchtrain: False, mm.dropouttrain: False}
    pos_pred_prob = sess.run(mm.y_pred_prob, feed_dict=feedDict)

    posX_ = pd.concat([pd.DataFrame(pos_pred_prob[:, 1], columns=['prob']), posX], axis=1)
    pos_aim = posX_.sort_values(by='prob').iloc[:pos_aimNum, 1:].reset_index(drop=True)

    # neg
    feedDict = {mm.tstx: negX, mm.Batchtrain: False, mm.dropouttrain: False}
    neg_pred_prob = sess.run(mm.y_pred_prob, feed_dict=feedDict)
    negX_ = pd.concat([pd.DataFrame(neg_pred_prob[:, 0], columns=['prob']), negX], axis=1)
    neg_aim = negX_.sort_values(by='prob').iloc[:neg_aimNum, 1:].reset_index(drop=True)

    Y = list(np.ones(int(len(pos_aim)))) + list(np.zeros(int(len(neg_aim))))
    Y = list(map(int, Y))

    X = pd.concat([pos_aim, neg_aim], axis=0, ignore_index=True)
    return X,Y


def train(datapre,
          params,
          Trainphotopath, model_self_name, ckptpath, lastckptpath):
    tf.reset_default_graph()
    with tf.Session() as trainSess:
        mm = mymodel(params)
        trainX, trainY, batchNum = GetBatchTask(params, datapre, mode='train')

        allnegx = pd.read_csv(datapre + 'class0_train_data.csv')
        allposx = pd.read_csv(datapre + 'class1_train_data.csv')



        tstPosX, tstPosY, tstnegX, tstnegY = GetYanzhengData_precModify(datapre)
        testx = pd.concat([tstPosX, tstnegX], axis=0, ignore_index=True)
        testy = pd.concat([pd.DataFrame(tstPosY), pd.DataFrame(tstnegY)], axis=0, ignore_index=True)

        PosX, PosY, negX, negY = GetTestData_precModify(dataprefix)
        realtestx = pd.concat([PosX, negX], axis=0, ignore_index=True)
        realtesty = pd.concat([pd.DataFrame(PosY), pd.DataFrame(negY)], axis=0, ignore_index=True)


        # print('please choose a mode: (input 1 or 2)')
        # Inp = input('1--frozen Feature extractor, only train DN?    or     2--train DN and pre-trained feature extractor together?\n').lower()
        # if Inp == '1':
        #     print('loading pre-trained feature extractor model....', lastckptpath)
        #     var_list = [var for var in tf.trainable_variables() if 'Mlp_embedding' in var.name]
        #     saver_pre = tf.train.Saver(var_list)
        #     trainSess.run(tf.global_variables_initializer())
        #     saver_pre.restore(trainSess, lastckptpath)
        # elif Inp == '2':
        #     # train feature extractor and DN together, but remember to unannotated the
        # '''Fine-tune : unfrozen Feature extractor, train feature extractor and DN together''' part in second_Stage_classDefine.py
        # and annotated the
        # '''Only train DN: frozen Feature extractor, only train the DN part''' part in second_Stage_classDefine.py
        #     print('loading last model....', lastckptpath)
        #     var_list = [var for var in tf.trainable_variables() if 'cls_optimizer' not in var.name]
        #     saver_pre = tf.train.Saver(var_list)
        #     trainSess.run(tf.global_variables_initializer())
        #     saver_pre.restore(trainSess, lastckptpath)
        # else:
        #     print('input error...')
        #     return -1
        print('loading pre-trained feature extractor model....', lastckptpath)
        print('frozen the pre-trained feature extractor and only train DN....')
        var_list = [var for var in tf.trainable_variables() if 'Mlp_embedding' in var.name]
        saver_pre = tf.train.Saver(var_list)
        trainSess.run(tf.global_variables_initializer())
        saver_pre.restore(trainSess, lastckptpath)

        for variable in tf.trainable_variables():
            print(variable)

        empha_epoch = batchNum // 5   # choose emphasizing 5 times for one epoch

        TrainLoss = []
        for _ in range(params.epochs):
            ephaX, ephaY = choostEmphasizeSet(trainSess, mm, allposx, allnegx, params)
            for _B in range(batchNum):
                train_x = trainX[_B]
                train_label = trainY[_B]

                feedDict = {mm.train_x: train_x,
                            mm.train_label: train_label,
                            mm.Batchtrain: True, mm.dropouttrain: True}

                trainlr, _, trainloss, global_step = trainSess.run([mm.only_train_clslearning_rate,
                                                                    mm.cls_optimizer,
                                                                    mm.only_train_cls_trainloss,
                                                                    mm.only_train_cls_glbstep],
                                                                   feed_dict=feedDict)
                if global_step % empha_epoch ==0:
                    # Emphasizing
                    feedDict = {mm.train_x: ephaX, mm.train_label: ephaY,
                                mm.Batchtrain: True, mm.dropouttrain: True}
                    _op = trainSess.run(mm.cls_optimizer, feed_dict=feedDict)
                    # If only train the DN with frozen feature extractor, and dont fine-tune
                    # no shuffle step needed here

                if global_step == 1 or global_step % 400 ==0:
                    print('===========================================================')
                    print(global_step, ' step', ' train loss: ', trainloss)
                    TrainLoss.append(trainloss)

                    print('------------ val result -------------')
                    feedDict = {mm.tstx: testx, mm.Batchtrain: False, mm.dropouttrain: False}
                    test_pred_prob = trainSess.run(mm.y_pred_prob, feed_dict=feedDict)  # np.darray
                    pr_score = average_precision_score(testy, test_pred_prob[:, 1])

                    tstPred = np.argmax(test_pred_prob, axis=-1)
                    confusemat = confusion_matrix(testy, tstPred)
                    classrpt = classification_report(testy, tstPred)
                    print(confusemat)
                    print(classrpt)
                    print("pr_score: ", pr_score)
                    TN = confusemat[0][0]
                    FP = confusemat[0][1]
                    FN = confusemat[1][0]
                    TP = confusemat[1][1]
                    GM = np.sqrt(((TP) / (TP + FP)) * (TP / (TP + FN)))
                    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 0.0000001)
                    print("GM: ", GM)
                    print("MCC: ", MCC)

                    print('------------ test result -------------')
                    feedDict = {mm.tstx: realtestx, mm.Batchtrain: False, mm.dropouttrain: False}
                    test_pred_prob = trainSess.run(mm.y_pred_prob, feed_dict=feedDict)
                    pr_score = average_precision_score(realtesty, test_pred_prob[:, 1])

                    tstPred = np.argmax(test_pred_prob, axis=-1)
                    confusemat = confusion_matrix(realtesty, tstPred)
                    classrpt = classification_report(realtesty, tstPred)
                    print(confusemat)
                    print(classrpt)
                    print("pr_score: ", pr_score)
                    TN = confusemat[0][0]
                    FP = confusemat[0][1]
                    FN = confusemat[1][0]
                    TP = confusemat[1][1]
                    GM = np.sqrt(((TP) / (TP + FP)) * (TP / (TP + FN)))
                    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 0.0000001)
                    print("GM: ", GM)
                    print("MCC: ", MCC)

                    sss = tf.train.Saver()
                    sss.save(trainSess, ckptpath+str(global_step) +'/')

        # Save Fig
        phototitle = 'train batch:' + str(batchNum) + ' epochs:' + str(params.epochs) + \
                     '\n lr:' + str(params.lr)
        if params.decay_flag == 1:
            phototitle += '  decayStep: ' + str(params.decay_steps) + \
                          "  decayRate: " + str(params.decay_rate)
        else:
            phototitle += '  noDecay '

        plt.title(phototitle)
        plt.plot(TrainLoss, color='red', linestyle='solid', label='Train Loss')
        plt.legend()
        plt.xlabel('times')
        plt.ylabel('loss')
        plt.savefig(Trainphotopath + model_self_name + '.jpg')
        plt.show()


def get_data(dprefix, params, mode='train'):
    if mode == 'train':
        # train
        # mini-BGD
        print('get train data ...')
        trainX, trainY, batchNum = GetBatchTask(params, dprefix, mode=mode)
        return trainX, trainY, batchNum

    elif mode == 'train_test' :
        #  test
        print('get train test data ...')
        # in test process
        trainX, trainY, batchNum = GetBatchTask(params, dataprefix, mode=mode)
        return trainX, trainY, batchNum





if __name__ == '__main__':
    params = Params()

    '''
        ####################### path deifintion ###############################
    '''

    SavePrefix = create_path('./SaveModel/credit/twoStage/')
    dataprefix = './creditFraud/'


    if params.decay_flag == 0:
        model_self_name = params.modelname + 'noDecay'
    elif params.decay_flag == 1:
        model_self_name = params.modelname + 'hasDecay'


    Trainphotopath = create_path(SavePrefix  + '/Model/'  + '/TrainPng/')
    ckptpath = create_path(SavePrefix + '/Model/' + model_self_name + '/')
    # chosen model of the pre-trained feature extractor in first stage
    lastckptpath = SavePrefix + '/Model/' + params.lastmodelname + '/15200/'
    print('this model is ', model_self_name)



    train(dataprefix, params, Trainphotopath, model_self_name, ckptpath, lastckptpath)






















