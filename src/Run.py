import numpy as np
from AutoEncoder import AutoEncoder
from KNNClassifier import KNNClassifier
from MyPCA import MyPCA
from Preprocessing import Preprocessing
from MySom import MySom, translateToCoords

from RunHelper import Method, GetMethodIndex, GetProjType, GetAeType, GetGridSize, GetNumOfPrinComp, \
    GetSlideDivisor, GetSleepIndicator, GetNumOfAeSplits, GetNumOfSomSplits, GetStartingGridSize, \
    GetEndingGridSize, GetStartingAeType, GetEndingAeType, GetStartingPca, GetEndingPca, GetLogFilePath

runVal = GetMethodIndex()
if runVal == "1":
    method = Method.SingleEncoder
    aeTypeStart = GetStartingAeType()
    aeTypeEnd = GetEndingAeType()
elif runVal == "2":
    method = Method.SingleSom
    numOfSomSplits = GetNumOfSomSplits()
    slideDiv = GetSlideDivisor()
    coordBasedVal = GetProjType()
    gridSizeStart = GetStartingGridSize()
    gridSizeEnd = GetEndingGridSize()
    if coordBasedVal == "1": isCoordBased = True
    else: isCoordBased = False
elif runVal == "3":
    method = Method.EncoderPlusSom
    aeType = GetAeType()
    gridSize = GetGridSize()
    numOfSomSplits = GetNumOfSomSplits()
    coordBasedVal = GetProjType()
    if coordBasedVal == "2": isCoordBased = False
elif runVal == "4":
    method = Method.MultipleEncoders
    numOfAeSplits = GetNumOfAeSplits()
    aeTypeStart = GetStartingAeType()
    aeTypeEnd = GetEndingAeType()
elif runVal == "5":
    method = Method.MultipleSoms
    numOfSomSplits = GetNumOfSomSplits()
    slideDiv = GetSlideDivisor()
    gridSizeStart = GetStartingGridSize()
    gridSizeEnd = GetEndingGridSize()
elif runVal == "6":
    method = Method.PCA
    pcaStart = GetStartingPca()
    pcaEnd = GetEndingPca()
elif runVal == "7":
    method = Method.PcaPlusEncoder
    numOfPrinComp = GetNumOfPrinComp()
    aeType = GetAeType()
elif runVal == "8":
    method = Method.PcaPlusSom
    numOfPrinComp = GetNumOfPrinComp()
    numOfSomSplits = GetNumOfSomSplits()
elif runVal == "9":
    method = Method.MultipleEncodersAndSOMs
    aeType = GetAeType()
    numOfAeSplits = GetNumOfAeSplits()
    numOfSomSplits = GetNumOfSomSplits()

sleepIndicator = GetSleepIndicator()

# Set this to the directory where TrainingSet.csv and TestingSet.csv are
baseDir = "C:/Dev/DataSets/ICBEB/Raw"
########################################################################

# Initialise training, validation and testing data
trainX, trainY = Preprocessing.LoadAllSamplesFromCsv(baseDir + "/TrainingSet.csv", True)
testX, testY = Preprocessing.LoadAllSamplesFromCsv(baseDir + "/TestingSet.csv", True)
logFileDir = baseDir + "/KNN Results/"

# For POC purposes######
#trainX = trainX[0:1]
#trainY = trainY[0:1]
#testX = testX[0:1]
#testY = testY[0:1]
########################

def runKNN(trainX, testX, trainY, testY, logFilePath):
    # Initialise, train and test KNN. Use 3-fold-cross validation. Grid search k values 1-9.
    knn = KNNClassifier(10, 3, logFilePath)
    knn.train(trainX, trainY)
    knn.test(testX, testY)

def runAutoEncoder(autoEncoderType, logFilePath, trainX, testX):
    # Initialise, train and test AutoEncoder
    autoEncoder = AutoEncoder(trainX.shape[1], autoEncoderType, logFilePath)
    autoEncoder.train(trainX, 256, 100)
    encodedTrainX = autoEncoder.encode(trainX, True)  # This is what we pass to our SOM or KNN to train
    encodedTestX = autoEncoder.encode(testX, False)  # This is what we pass to out SOM or KNN to test

    return encodedTrainX, encodedTestX

def runMultipleEncoder(numOfSplits, aeType, logFilePath, splitByIndexTrainX, splitByIndexTestX):
    for i in range(numOfSplits):
        nextTrainX, nextTestX = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                               trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i])
        if i == 0:
            newTrainX = nextTrainX
            newTestX = nextTestX
        else:  # merge split encodings
            newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
            newTestX = np.concatenate((newTestX, nextTestX), axis=1)

    return newTrainX, newTestX

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

def runMultipleSom(numOfSplits, gridSize, logFilePath, splitByIndexTrainX, splitByIndexTestX, trainX, testX):
    for i in range(numOfSplits):
        nextTrainX, nextTestX = runSom(gridSize=gridSize, somSplit=1, logFilePath=logFilePath,
                                       trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i],
                                       originalTrainSize=trainX.shape[0], originalTestSize=testX.shape[0])
        if i == 0:
            newTrainX = nextTrainX
            newTestX = nextTestX
        else:  # merge split projections
            newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
            newTestX = np.concatenate((newTestX, nextTestX), axis=1)

    return newTrainX, newTestX


if method == Method.EncoderPlusSom:
    logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, autoEncoderType=aeType, somGridSize=gridSize,
                                 numOfSomSplits=numOfSomSplits, numOfInputDim=trainX.shape[1])

    encodedTrainX, encodedTestX = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                                 trainX=trainX, testX=testX)

    windowSize = int(encodedTrainX.shape[1] / numOfSomSplits)
    splitTrainX = Preprocessing.SlidingWindowSplitter(dataX=trainX, windowSize=windowSize, slide=windowSize)
    splitTestX = Preprocessing.SlidingWindowSplitter(dataX=testX, windowSize=windowSize, slide=windowSize)

    encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSomSplits, logFilePath=logFilePath,
                                         trainX=splitTrainX, testX=splitTestX, originalTrainSize=encodedTrainX.shape[0],
                                         originalTestSize=encodedTestX.shape[0])

    runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.SingleEncoder:
    for aeType in range(aeTypeStart, aeTypeEnd):
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, autoEncoderType=aeType,
                                     numOfInputDim=trainX.shape[1])

        newTrainX, newTestX = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                             trainX=trainX, testX=testX)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.MultipleEncoders:
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(dataX=trainX, numOfSplits=numOfAeSplits, slideDivisor=1)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(dataX=testX, numOfSplits=numOfAeSplits, slideDivisor=1)

    for aeType in range(aeTypeStart, aeTypeEnd):
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, autoEncoderType=aeType,
                                     numOfAeSplits=numOfAeSplits, numOfInputDim=trainX.shape[1]/numOfAeSplits)

        newTrainX, newTestX = runMultipleEncoder(numOfAeSplits, aeType, logFilePath,
                                                 splitByIndexTrainX, splitByIndexTestX)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.SingleSom:
    windowSize = int(trainX.shape[1] / numOfSomSplits)
    splitTrainX = Preprocessing.SlidingWindowSplitter(dataX=trainX, windowSize=windowSize, slide=windowSize / slideDiv)
    splitTestX = Preprocessing.SlidingWindowSplitter(dataX=testX, windowSize=windowSize, slide=windowSize / slideDiv)
    numOfSplits = numOfSomSplits * slideDiv - (slideDiv - 1)

    for gridSize in range(gridSizeStart, gridSizeEnd, 5):
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, somGridSize=gridSize,
                                     numOfSomSplits=numOfSplits)

        encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSplits, logFilePath=logFilePath,
                                             trainX=splitTrainX, testX=splitTestX, originalTrainSize=trainX.shape[0],
                                             originalTestSize=testX.shape[0])

        runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.MultipleSoms:
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(dataX=trainX, numOfSplits=numOfSomSplits, slideDivisor=slideDiv)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(dataX=testX, numOfSplits=numOfSomSplits, slideDivisor=slideDiv)
    numOfSplits = numOfSomSplits * slideDiv - (slideDiv - 1)

    for gridSize in range(gridSizeStart, gridSizeEnd, 5):
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, somGridSize=gridSize,
                                     numOfSomSplits=numOfSplits)

        newTrainX, newTestX = runMultipleSom(numOfSplits, gridSize, logFilePath, splitByIndexTrainX,
                                             splitByIndexTestX, trainX, testX)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PCA:
    for i in range(pcaStart, pcaEnd):
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, numOfPcaComp=i)

        myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, i, logFilePath) # newTrainX = principal components
        newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PcaPlusEncoder:
    logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, autoEncoderType=aeType,
                                       numOfPcaComp=numOfPrinComp, numOfInputDim=numOfPrinComp)

    myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp, logFilePath)  # newTrainX = principal components
    newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

    newTrainX, newTestX = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                         trainX=newTrainX, testX = newTestX)

    runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PcaPlusSom:
    gridSize = 40
    logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, somGridSize=gridSize,
                                 numOfSomSplits=numOfSomSplits, numOfPcaComp=numOfPrinComp)

    myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp, logFilePath)  # newTrainX = principal components
    newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

    windowSize = int(newTrainX.shape[1] / numOfSomSplits)
    splitTrainX = Preprocessing.SlidingWindowSplitter(dataX=trainX, windowSize=windowSize, slide=windowSize)
    splitTestX = Preprocessing.SlidingWindowSplitter(dataX=testX, windowSize=windowSize, slide=windowSize)

    encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSomSplits, isCoordBased=isCoordBased,
                                         logFilePath=logFilePath, trainX=splitTrainX, testX=splitTestX,
                                         originalTrainSize=newTrainX.shape[0], originalTestSize=newTestX.shape[0])
    runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.MultipleEncodersAndSOMs:
    gridSize = 40
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfAeSplits, slideDivisor=1)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfAeSplits, slideDivisor=1)

    logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, autoEncoderType=aeType, somGridSize=gridSize,
                                 numOfSomSplits=numOfSomSplits, numOfAeSplits=numOfAeSplits,
                                 numOfInputDim=trainX.shape[1] / numOfAeSplits)

    newTrainX, newTestX = runMultipleEncoder(numOfAeSplits, aeType, logFilePath,
                                             splitByIndexTrainX, splitByIndexTestX)

    splitByIndexTrainX = Preprocessing.SplitDataByIndex(newTrainX, numOfSomSplits, slideDivisor=1)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(newTestX, numOfSomSplits, slideDivisor=1)

    newTrainX, newTestX = runMultipleSom(numOfSomSplits, gridSize, logFilePath, splitByIndexTrainX,
                                         splitByIndexTestX, trainX, testX)

    runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

if sleepIndicator == 1:
    import os
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")