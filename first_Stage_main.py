#coding=utf-8
#__author__='YHR'

from first_Stage_ClassDefine import Mulfeat_ce_sia as mymodel
from first_Stage_Params import Params
from getData import GetBatchTask, GetTestData_precModify, GetYanzhengData_precModify
from helpFunc import create_path
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


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

    neg_y = pd.DataFrame(list(map(int, list(np.zeros(int(len(neg_aim)))))), columns=['class'])
    pos_y = pd.DataFrame(list(map(int, list(np.ones(int(len(pos_aim)))))), columns=['class'])

    negAll = pd.concat([neg_y, neg_aim], axis=1, ignore_index=True)
    posAll = pd.concat([pos_y, pos_aim], axis=1, ignore_index=True)

    x0yneed = negAll.iloc[:(params.batch_samples // 2), :].reset_index(drop=True)

    x0ygive = negAll.iloc[(params.batch_samples // 2):, :].reset_index(drop=True)

    x1y = pd.concat([posAll, x0ygive], axis=0, ignore_index=True).reset_index(drop=True)

    x0 = x0yneed.iloc[:, 1:]
    y0 = x0yneed.iloc[:, 0].ravel()

    x1 = x1y.iloc[:, 1:]
    y1 = x1y.iloc[:, 0].ravel()

    x0equalx1 = (y0 == y1).astype(float)

    return x0, y0, x1, y1, x0equalx1


def train(params,
          datapre,
          Trainphotopath, model_self_name, ckptpath):
    tf.reset_default_graph()
    Trainx, Trainy, batchNum = GetBatchTask(params, datapre, mode='train')
    allnegx = pd.read_csv(datapre + 'class0_train_data.csv')
    allposx = pd.read_csv(datapre + 'class1_train_data.csv')

    tstPosX, tstPosY, tstnegX, tstnegY = GetYanzhengData_precModify(datapre)
    testx = pd.concat([tstPosX, tstnegX], axis=0, ignore_index=True)
    testy = pd.concat([pd.DataFrame(tstPosY), pd.DataFrame(tstnegY)] ,axis=0, ignore_index=True)

    PosX, PosY, negX, negY = GetTestData_precModify(dataprefix)
    realtestx = pd.concat([PosX, negX], axis=0, ignore_index=True)
    realtesty = pd.concat([pd.DataFrame(PosY), pd.DataFrame(negY)], axis=0, ignore_index=True)

    empha_epoch = batchNum//5   # choose emphasizing 5 times for one epoch

    with tf.Session() as trainSess:
        mm = mymodel(params)
        print('train a new model....')
        trainSess.run(tf.global_variables_initializer())

        for variable in tf.trainable_variables():
            print(variable)

        TrainLoss = []
        Train_clsloss = []
        for _epo in range(params.epochs):
            ep_x0, ep_y0, ep_x1, ep_y1, ep_x0equalx1 = choostEmphasizeSet(trainSess, mm, allposx, allnegx, params)
            for _b in range(batchNum):

                allx = pd.DataFrame(Trainx[_b]).reset_index(drop=True)
                ally = pd.DataFrame(Trainy[_b]).reset_index(drop=True)
                ally.columns = ['class']

                alldata = pd.concat([ally, allx], axis=1).sample(frac=1, random_state=666).reset_index(drop=True)

                train_x0 = alldata.iloc[:(params.batch_samples//2), 1:]
                train_y0 = alldata.iloc[:(params.batch_samples//2), 0].ravel()

                train_x1 = alldata.iloc[(params.batch_samples // 2):, 1:]
                train_y1 = alldata.iloc[(params.batch_samples // 2):, 0].ravel()

                x0equalx1 = (train_y0 == train_y1).astype(float)


                feedDict = {mm.train_x0: train_x0, mm.train_y0: train_y0,
                            mm.train_x1: train_x1, mm.train_y1: train_y1,
                            mm.x0equalx1: x0equalx1,
                            mm.Batchtrain: True, mm.dropouttrain: True}


                _op, trainloss, cls_loss, sia_loss,  global_step, train_lr = trainSess.run(
                    [mm.optimizer,
                     mm.trainLoss, mm.ce_loss, mm.sia_loss,
                     mm.global_step, mm.learning_rate],
                    feed_dict=feedDict)


                if global_step %  empha_epoch ==0:
                    # Emphasizing step
                    print(global_step, ' epoch,  Emphasizing.....')
                    feedDict = {mm.train_x0: ep_x0, mm.train_y0: ep_y0,
                                mm.train_x1: ep_x1, mm.train_y1: ep_y1,
                                mm.x0equalx1: ep_x0equalx1,
                                mm.Batchtrain: True, mm.dropouttrain: True}
                    _op = trainSess.run(mm.optimizer, feed_dict=feedDict)

                    # shuffled to produce different pairs for next training process
                    ep_x = pd.concat([pd.DataFrame(ep_x0), pd.DataFrame(ep_x1)] , axis=0, ignore_index=True)
                    ep_y = pd.concat([pd.DataFrame(ep_y0), pd.DataFrame(ep_y1)] , axis=0, ignore_index=True)
                    ep_xy = pd.concat([ep_y, ep_x], axis=1, ignore_index=True).sample(frac=1, replace=False, random_state=global_step)
                    ep_xy_up = ep_xy.iloc[:(params.batch_samples // 2), :].reset_index(drop=True)

                    ep_xy_down = ep_xy.iloc[(params.batch_samples // 2):, :].reset_index(drop=True)

                    ep_x0 = ep_xy_up.iloc[:, 1:]
                    ep_y0 = ep_xy_up.iloc[:, 0].ravel()

                    ep_x1 = ep_xy_down.iloc[:, 1:]
                    ep_y1 = ep_xy_down.iloc[:, 0].ravel()

                    ep_x0equalx1 = (ep_y0 == ep_y1).astype(float)


                if global_step == 1 or global_step % 400 == 0:
                    print('===========================================================')
                    print(global_step, ' 轮', 'lr ', train_lr, ' train loss: ', trainloss,
                          ' focal loss', cls_loss, ' sia_loss', sia_loss)

                    TrainLoss.append(trainloss)
                    Train_clsloss.append(cls_loss)

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
                    test_pred_prob = trainSess.run(mm.y_pred_prob, feed_dict=feedDict)  # np.darray
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

                    if global_step > 10000:
                        # for saving space, only the models with training steps more than 10000 steps will be saved
                        # but you can choose to save all models.
                        # choose the best model both in val and test to be the pre-trained model for two stage
                        sa = tf.train.Saver()
                        sa.save(trainSess, ckptpath + str(global_step) + '/')
                        print('save finish...')

        # Save Fig
        phototitle = 'train batchNum:' + str(batchNum) + ' epochs:' + str(params.epochs) + \
                     '\n lr:' + str(params.lr)
        if params.decay_flag == 1:
            phototitle += '  decayStep: ' + str(params.decay_steps) + \
                          "  decayRate: " + str(params.decay_rate)
        else:
            phototitle += '  noDecay '

        plt.title(phototitle)
        plt.plot(TrainLoss, color='red', linestyle='solid', label='Train Loss')
        plt.plot(Train_clsloss, color='blue', linestyle='dotted', label='Train focal Loss')
        plt.legend()
        plt.xlabel('times')
        plt.ylabel('loss')
        plt.savefig(Trainphotopath + model_self_name + '.png')
        plt.show()



if __name__ == "__main__":
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

    Trainphotopath = create_path(SavePrefix + '/Model/' + '/TrainPng/')
    ckptpath = create_path(SavePrefix + '/Model/' + model_self_name + '/')
    print('this model is ', model_self_name)

    train(params, dataprefix, Trainphotopath, model_self_name, ckptpath)







