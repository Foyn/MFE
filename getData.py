#coding=utf-8
#__author__='YHR'
import pandas as pd
import numpy as np

def CopyDataToyouLike(Data, copyTimes):
    '''
    copyTimes : 打乱多少次
    最终数据为 1+ copy
    '''
    # 至少本身打乱一次
    Data_concat = Data.sample(frac=1, replace=False, random_state=1)

    for _ in range(copyTimes):
        shuffleData = Data.sample(frac=1, replace=False, random_state=_+20)  # 改变随机数种子
        Data_concat = pd.concat([Data_concat, shuffleData], axis=0, ignore_index=True)
    return Data_concat


def GetBatchTask(params, datapre, mode='train'):
    if mode == 'train':
        print("choost batch task dataset....")

        class0Data = pd.read_csv(datapre + '/class0_train_data.csv')
        class1Data = pd.read_csv(datapre + '/class1_train_data.csv')


        copyTimes = 10
        class1Data = CopyDataToyouLike(class1Data, copyTimes)



    print(len(class0Data), len(class1Data))
    # // normal part
    Y = list(np.ones(int(len(class1Data)))) + list(np.zeros(int(len(class0Data))))
    Y = pd.DataFrame(list(map(int, Y)))

    X = pd.concat([class1Data, class0Data], axis=0, ignore_index=True)

    Data = pd.concat([Y, X], axis=1)
    Data = Data.sample(frac=1, replace=False, random_state=999) # shuffle


    batches = params.batch_samples  # 256个/batch
    batchNum = len(Data) // batches

    trainX = []
    trainY = []

    for _ in range(batchNum):
        data = Data.iloc[_*batches: (_+1)*batches, :]
        trainX.append(data.iloc[:, 1:])
        trainY.append(data.iloc[:, 0])
    print(batchNum, batches)
    return trainX, trainY, batchNum





def GetYanzhengData_precModify(datapre):

    class0Data = pd.read_csv(datapre + '/class0_val_data.csv')
    class1Data = pd.read_csv(datapre + '/class1_val_data.csv')




    PosX = class1Data;  PosY = list(map(int, list(np.ones(int(len(class1Data))))))
    negX = class0Data;  negY = list(map(int, list(np.zeros(int(len(class0Data))))))
    print(len(negX), len(PosX))

    return PosX, PosY, negX, negY






def GetTestData_precModify(datapre):
    class0Data = pd.read_csv(datapre + '/class0_test_data.csv')
    class1Data = pd.read_csv(datapre + '/class1_test_data.csv')



    PosX = class1Data;  PosY = list(map(int, list(np.ones(int(len(class1Data))))))
    negX = class0Data;  negY = list(map(int, list(np.zeros(int(len(class0Data))))))

    return PosX, PosY, negX, negY

