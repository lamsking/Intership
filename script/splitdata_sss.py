#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:45:10 2019

@author: inra-cirad
"""

import csv
import pandas
import math
import numpy as np


def readFile(dataFile, columns):
    f = open(dataFile, 'r')
    dataList = []
    for x in f:
        x = x.strip('\n')
        ar = x.split(' ')
        dataList.append(ar)
    df = pandas.DataFrame(np.asarray(dataList), index=None, columns=columns)
    return df


def visualizeData(dataFile, columns):
    print(readFile(dataFile, columns))


def minkowski(a, b, r=1):
    sum = 0
    for i in range(len(a)):
        sum = sum + (abs(a[i] - b[i])**(1/r))
    return sum


def DistTrainDataTestData(trainFile, testFile, columns, dataColumns):
    train = readFile(trainFile, columns)
    trainNumeric = train[dataColumns].apply(pandas.to_numeric)
    trainList = trainNumeric.values.tolist()
    test = readFile(testFile, columns)
    testNumeric = test[dataColumns].apply(pandas.to_numeric)
    testList = testNumeric.values.tolist()
    for testIndex in range(len(testList)):
        listDistance = []
        for trainIndex in range(len(trainList)):
            listDistance.append(minkowski(testList[testIndex], trainList[trainIndex]))
        train['dist_t'+str(testIndex+1)] = pandas.Series(listDistance, index=None)

    return train, test


def knnClaissifyer(distDataFrame, testFrame, k=1):
    classes = []
    for j in range(len(testFrame)):
        listtrie = distDataFrame.sort_values(by=['dist_t'+str(j+1)])
        list  = []
        for i in range(k):
            list.append(listtrie.iloc[i][4])
        classes.append(voteClass(list))
    classes = [int(x) for x in classes]
    return classes


def voteClass(list):
    return max(list, key=list.count)


def setClassToTest(testFrame, listClasses):
    testFrame['class predicted'] = pandas.Series(listClasses, index=None)
    return testFrame

def precision(datatest):
    columnClasses = datatest[['class','class predicted']]
    numericColumnClasses = columnClasses.apply(pandas.to_numeric)
    listColumnClasses = numericColumnClasses.values.tolist()
    TP = 0
    for i in range(len(listColumnClasses)):
        #print(listColumnClasses[i][0], listColumnClasses[i][1])
        if listColumnClasses[i][0] == listColumnClasses[i][1]:
            TP = TP+1
    #print(TP)
    #print(len(listColumnClasses) - TP)
    return TP/ len(listColumnClasses)

def searchBestK(train, test ):
    i = 0
    maxPrecision = 0
    k = 2 * i + 1
    while k < len(test):
        classes = knnClaissifyer(train,test, k)
        prec = precision(setClassToTest(test, classes))
        print(prec)
        if prec >= maxPrecision:
            testclass = setClassToTest(test, classes)
            maxPrecision = prec
            bestK = k
            bestClass = testclass

        i = i + 1
        k = 2 * i + 1

    return bestClass, bestK, maxPrecision

def confusionMat(testFile, L=1, K=3):
    pass

"""
part : the number of part you will like to first retreive
partition: the number of partitions you will like to have 
"""

def splitDataFrame(trainFile, columns, dataColumns, part=2, partition=3):
    dataFrame = readFile(trainFile, columns)
    size = len(dataFrame)*part//partition
    train = dataFrame[:size].copy()
    evaluation = dataFrame[size:].copy()
    evaluation = evaluation.reset_index(drop=True)
    trainNumeric = train[dataColumns].apply(pandas.to_numeric)
    trainList = trainNumeric.values.tolist()
    print(len(trainList))
    evaluationNumeric = evaluation[dataColumns].apply(pandas.to_numeric)
    evaluationList = evaluationNumeric.values.tolist()
    print(len(evaluationList))
    for evaluationIndex in range(len(evaluationList)):
        listDistance = []
        for trainIndex in range(len(trainList)):
            listDistance.append(minkowski(evaluationList[evaluationIndex], trainList[trainIndex]))
        train['dist_t' + str(evaluationIndex + 1)] = pandas.Series(listDistance, index=None)

    return train, evaluation

#visualizeData("../data/iris/iris.tst", ['x1', 'x2', 'x3', 'x4', 'class'])
#visualizeData("../data/letter/let.tst", ['x1', 'x2', 'x3', 'x4','x5','x6','x7','x8', 'x9', 'x10', 'x11','x12','x13','x14','x15','x16', 'class'])

#DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst")
#train, test = DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst", ['x1', 'x2', 'x3', 'x4', 'class'], ['x1', 'x2', 'x3', 'x4'])
train, evaluation = splitDataFrame("../data/iris/iris.trn", ['x1', 'x2', 'x3', 'x4', 'class'], ['x1', 'x2', 'x3', 'x4'])

print(train)
print(evaluation)

bestclass, k, p = searchBestK(train, evaluation)

print(bestclass)
print("Meilleur k :", k, "Meilleur precesion :", p)
#classes = knnClaissifyer(train,5)
#print("******************")
#print(classes)
#print("******************")
#setClassToTest(test, classes)
#print(precision(setClassToTest(test, classes)))

#visualizeData("../data/iris/iris.tst", ['x1', 'x2', 'x3', 'x4', 'class'])
#print(len(readFile("../data/iris/iris.trn")))

exit(0)
