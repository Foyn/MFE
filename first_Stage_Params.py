#coding=utf-8
#__author__='YHR'
import argparse


def Params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_samples', default=256, type=int)  # credit 256,  other datasets 512

    # network params definition
    # feature dimension
    parser.add_argument('--nowDim', default=29, type=int)  # credit 29

    # training process params definition
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--decay_flag', default=1, type=int)
    parser.add_argument('--decay_steps', default=10000, type=int)
    parser.add_argument('--decay_rate', default=0.1, type=float)

    parser.add_argument('--batchNorm', default=1, type=int)
    parser.add_argument('--activation', default='relu', type=str)  # leakyrelu / relu


    # dropout option
    parser.add_argument('--dropout_option', default='usedropout', type=str)  # usedropout /nodropout
    parser.add_argument('--dropout_rate', default=0.1, type=float)


    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--modelname', default='FirstStage_3layerFeat_1_', type=str)
    # 64-32-16: 3

    return parser.parse_args()











