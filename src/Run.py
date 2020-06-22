import numpy as np
import RunNameHelper
from AutoEncoder import AutoEncoder
from KNNClassifier import KNNClassifier
from MyPCA import MyPCA
from Preprocessing import Preprocessing
from MySom import MySom, translateToCoords

from RunHelper import SampleType, Method, GetNumOfSplits, GetMethodIndex, GetProjType, GetAeType, GetGridSize, \
    GetNumOfPrinComp

baseDir = "C:/Dev/DataSets/PhysioNet/"
normalise, isCoordBased = False, True

print("1. Use raw samples\n2. Use filtered samples\n3. Use normalised samples")
rawSamplesVal = input()
if rawSamplesVal == "1": sampleType = SampleType.Raw
elif rawSamplesVal == "2": sampleType = SampleType.Filtered
elif rawSamplesVal == "3": normalise = True

runVal = GetMethodIndex()
if runVal == "1": method = Method.SingleEncoder
elif runVal == "2":
    method = Method.SingleSom
    numOfSplits = GetNumOfSplits()
    coordBasedVal = GetProjType()
    if coordBasedVal == "2": isCoordBased = False
elif runVal == "3": method = Method.EncoderPlusSom
elif runVal == "4":
    method = Method.MultipleEncoders
    numOfSplits = GetNumOfSplits()
elif runVal == "5":
    method = Method.MultipleSoms
    numOfSplits = GetNumOfSplits()
elif runVal == "6":
    method = Method.PCA
elif runVal == "7":
    method = Method.PcaPlusEncoder
    numOfPrinComp = GetNumOfPrinComp()
    aeType = GetAeType()
elif runVal == "8":
    method = Method.PcaPlusSom
    numOfPrinComp = GetNumOfPrinComp()
    numOfSplits = GetNumOfSplits()

if method == Method.EncoderPlusSom:
    aeType = GetAeType()
    gridSize = GetGridSize()
    numOfSplits = GetNumOfSplits()
    coordBasedVal = GetProjType()
    if coordBasedVal == "2": isCoordBased = False

# Initialise training, validation and testing data
trainX, trainY = Preprocessing.LoadAllSamplesFromCsv(baseDir + sampleType.name + "/TrainingSet.csv", True, 181)

testX, testY = Preprocessing.LoadAllSamplesFromCsv(baseDir + sampleType.name + "/TestingSet.csv", True, 181)
logFileDir = baseDir + sampleType.name + "/KNN Results/"
aeModelDir = baseDir + sampleType.name + "/Models/"

if normalise:
    trainX, testX = Preprocessing.NormaliseData(trainX, testX)

# For POC purposes######
#trainX = trainX[0:2000]
#trainY = trainY[0:2000]
#testX = testX[0:100]
#testY = testY[0:100]
########################

def runKNN(trainX, testX, trainY, testY, logFilePath):
    # Initialise, train and test KNN. Use 3-fold-cross validation. Grid search k values 1-9.
    knn = KNNClassifier(10, 3, logFilePath)
    knn.train(trainX, trainY)
    knn.test(testX, testY)

def runAutoEncoder(autoEncoderType, runName, logFilePath, trainX, testX):
    # Initialise, train and test AutoEncoder
    autoEncoder = AutoEncoder(trainX.shape[1], autoEncoderType, logFilePath)
    autoEncoder.train(trainX, 256, 100)
    encodedTrainX = autoEncoder.encode(trainX, True)  # This is what we pass to our SOM or KNN to train
    encodedTestX = autoEncoder.encode(testX, False)  # This is what we pass to out SOM or KNN to test

    return encodedTrainX, encodedTestX

def runSom(gridSize, somSplit, logFilePath, trainX, testX, originalTrainSize, originalTestSize,
           isCoordBased=True, isMultipleSom=False):
    mySom = MySom(trainX, gridSize, logFilePath)
    mySom.train(100)

    projectedTrainX = mySom.project(trainX, isTrainData=True)
    projectedTestX = mySom.project(testX, isTrainData=False)

    if isMultipleSom:
        projectedTrainX = np.reshape(projectedTrainX, (projectedTrainX.shape[0], somSplit))
        projectedTestX = np.reshape(projectedTestX, (projectedTestX.shape[0], somSplit))
    else:
        projectedTrainX = np.reshape(projectedTrainX, (originalTrainSize, somSplit))
        projectedTestX = np.reshape(projectedTestX, (originalTestSize, somSplit))

    if isCoordBased:
        encodedTrainX = translateToCoords(projectedTrainX, gridSize)  # Pass to our KNN to train
        encodedTestX = translateToCoords(projectedTestX, gridSize)  # Pass to our KNN to test
    else:
        return projectedTrainX, projectedTestX

    return encodedTrainX, encodedTestX


if method == Method.EncoderPlusSom:
    runName = RunNameHelper.GetRunName(method, autoEncoderType=aeType, somGridSize=gridSize,
                                       numOfSplits=numOfSplits, normalise=normalise)

    encodedTrainX, encodedTestX = runAutoEncoder(True, autoEncoderType=aeType, runName=runName,
                                                 trainX=trainX, testX=testX)

    # Example: (1000, 300) => (1000*somSplit, 300/somSplit) => (2000, 150)
    splitTrainX = np.reshape(encodedTrainX, (encodedTrainX.shape[0] * numOfSplits, int(encodedTrainX.shape[1] / numOfSplits)))
    splitTestX = np.reshape(encodedTestX, (encodedTestX.shape[0] * numOfSplits, int(encodedTestX.shape[1] / numOfSplits)))

    logFilePath = logFileDir + runName + ".txt"
    encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSplits, logFilePath=logFilePath,
                                         trainX=splitTrainX, testX=splitTestX, originalTrainSize=encodedTrainX.shape[0],
                                         originalTestSize=encodedTestX.shape[0])
    runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.SingleEncoder:
    for aeType in range(2, 5):
        runName = RunNameHelper.GetRunName(method, autoEncoderType=aeType)
        logFilePath = logFileDir + runName + ".txt"
        newTrainX, newTestX = runAutoEncoder(autoEncoderType=aeType, runName=runName, logFilePath=logFilePath,
                                             trainX=trainX, testX=testX)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.MultipleEncoders:
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfSplits)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfSplits)

    for aeType in range(3, 4):
        runName = RunNameHelper.GetRunName(autoEncoderType=aeType, numOfSplits=numOfSplits)
        logFilePath = logFileDir + runName + ".txt"

        for i in range(numOfSplits):
            nextTrainX, nextTestX = runAutoEncoder(autoEncoderType=aeType, runName=runName, logFilePath=logFilePath,
                                                   trainX=splitByIndexTrainX[i], testX = splitByIndexTestX[i])

            if i == 0:
                newTrainX = nextTrainX
                newTestX = nextTestX
            else:  # merge split encodings
                newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
                newTestX = np.concatenate((newTestX, nextTestX), axis=1)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.SingleSom:
    # Example: (1000, 300) => (1000*somSplit, 300/somSplit) => (2000, 150)
    splitTrainX = np.reshape(trainX, (trainX.shape[0] * numOfSplits, int(trainX.shape[1] / numOfSplits)))
    splitTestX = np.reshape(testX, (testX.shape[0] * numOfSplits, int(testX.shape[1] / numOfSplits)))

    for gridSize in range(5, 51, 5):
        runName = RunNameHelper.GetRunName(method, somGridSize=gridSize, numOfSplits=numOfSplits, normalise=normalise)
        logFilePath = logFileDir + runName + ".txt"
        encodedTrainX, encodedTestX = runSom(gridSize, numOfSplits, isCoordBased, logFilePath, splitTrainX, splitTestX, False)
        runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.MultipleSoms:
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfSplits)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfSplits)

    for gridSize in range(5, 51, 5):
        runName = RunNameHelper.GetRunName(method=method, somGridSize=gridSize, numOfSplits=numOfSplits, normalise=normalise)
        logFilePath = logFileDir + runName + ".txt"

        for i in range(numOfSplits):
            nextTrainX, nextTestX = runSom(gridSize=gridSize, somSplit=1, logFilePath=logFilePath,
                                           trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i])

            if i == 0:
                newTrainX = nextTrainX
                newTestX = nextTestX
            else: # merge split projections
                newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
                newTestX = np.concatenate((newTestX, nextTestX), axis=1)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PCA:
    for i in range(20, 33):
        runName = RunNameHelper.GetRunName(method, normalise=normalise, numOfPcaComp=i)
        logFilePath = logFileDir + runName + ".txt"
        myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, i, logFilePath) # newTrainX = principal components
        newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PcaPlusEncoder:
    runName = RunNameHelper.GetRunName(method=method, autoEncoderType=aeType, normalise=normalise,
                                       numOfPcaComp=numOfPrinComp)
    logFilePath = logFileDir + runName + ".txt"

    myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp, logFilePath)  # newTrainX = principal components
    newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

    newTrainX, newTestX = runAutoEncoder(autoEncoderType=aeType, runName=runName, logFilePath=logFilePath,
                                                   trainX=newTrainX, testX = newTestX)

    runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PcaPlusSom:
    gridSize = 40
    runName = RunNameHelper.GetRunName(method=method, somGridSize=gridSize, normalise=normalise,
                                       numOfSplits=numOfSplits, numOfPcaComp=numOfPrinComp)
    logFilePath = logFileDir + runName + ".txt"

    myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp, logFilePath)  # newTrainX = principal components
    newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

    splitTrainX = np.reshape(newTrainX, (newTrainX.shape[0] * numOfSplits, int(newTrainX.shape[1] / numOfSplits)))
    splitTestX = np.reshape(newTestX, (newTestX.shape[0] * numOfSplits, int(newTestX.shape[1] / numOfSplits)))

    encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSplits, isCoordBased=isCoordBased,
                                         logFilePath=logFilePath, trainX=splitTrainX, testX=splitTestX,
                                         originalTrainSize=newTrainX.shape[0], originalTestSize=newTestX.shape[0])
    runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

import os
os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")