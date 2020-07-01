import numpy as np
import RunNameHelper
from AutoEncoder import AutoEncoder
from KNNClassifier import KNNClassifier
from MyPCA import MyPCA
from Preprocessing import Preprocessing
from MySom import MySom, translateToCoords

from RunHelper import Method, GetMethodIndex, GetProjType, GetAeType, GetGridSize, GetNumOfPrinComp, \
    GetSlideDivisor, GetSleepIndicator, GetNumOfAeSplits, GetNumOfSomSplits, GetStartingGridSize, \
    GetEndingGridSize, GetStartingAeType, GetEndingAeType, GetStartingPca, GetEndingPca

runVal = GetMethodIndex()
if runVal == "1":
    method = Method.SingleEncoder
    aeTypeStart = GetStartingAeType()
    aeTypeEnd = GetEndingAeType()
elif runVal == "2":
    method = Method.SingleSom
    numOfSomSplits = GetNumOfSomSplits()
    slideDivisor = GetSlideDivisor()
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
    slideDivisor = GetSlideDivisor()
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
    slideDivisor = GetSlideDivisor()
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
    autoEncoder.train(trainX, 256, 1)
    encodedTrainX = autoEncoder.encode(trainX, True)  # This is what we pass to our SOM or KNN to train
    encodedTestX = autoEncoder.encode(testX, False)  # This is what we pass to out SOM or KNN to test

    return encodedTrainX, encodedTestX

def runSom(gridSize, somSplit, logFilePath, trainX, testX, originalTrainSize, originalTestSize,
           isCoordBased=True, isMultipleSom=False):
    mySom = MySom(trainX, gridSize, logFilePath)
    mySom.train(1)

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
                                       numOfSomSplits=numOfSomSplits, numOfInputDim=trainX.shape[1])
    logFilePath = logFileDir + runName + ".txt"

    encodedTrainX, encodedTestX = runAutoEncoder(autoEncoderType=aeType, runName=runName, logFilePath=logFilePath,
                                                 trainX=trainX, testX=testX)

    # Example: (1000, 300) => (1000*somSplit, 300/somSplit) => (2000, 150)
    splitTrainX = np.reshape(encodedTrainX, (encodedTrainX.shape[0] * numOfSomSplits,
                                             int(encodedTrainX.shape[1] / numOfSomSplits)))
    splitTestX = np.reshape(encodedTestX, (encodedTestX.shape[0] * numOfSomSplits,
                                           int(encodedTestX.shape[1] / numOfSomSplits)))

    logFilePath = logFileDir + runName + ".txt"
    encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSomSplits, logFilePath=logFilePath,
                                         trainX=splitTrainX, testX=splitTestX, originalTrainSize=encodedTrainX.shape[0],
                                         originalTestSize=encodedTestX.shape[0])
    runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.SingleEncoder:
    for aeType in range(aeTypeStart, aeTypeEnd):
        runName = RunNameHelper.GetRunName(method, autoEncoderType=aeType, numOfInputDim=trainX.shape[1])
        logFilePath = logFileDir + runName + ".txt"
        newTrainX, newTestX = runAutoEncoder(autoEncoderType=aeType, runName=runName, logFilePath=logFilePath,
                                             trainX=trainX, testX=testX)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.MultipleEncoders:
    slideDivisor = 2
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfAeSplits, slideDivisor=slideDivisor)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfAeSplits, slideDivisor=slideDivisor)
    numOfSplits = numOfAeSplits * slideDivisor - (slideDivisor - 1)

    for aeType in range(aeTypeStart, aeTypeEnd):
        runName = RunNameHelper.GetRunName(method, autoEncoderType=aeType, numOfSomSplits=numOfSplits,
                                           numOfInputDim=trainX.shape[1]/numOfSplits)
        logFilePath = logFileDir + runName + ".txt"

        for i in range(numOfSplits):
            nextTrainX, nextTestX = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
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
    windowSize = int(trainX.shape[1] / numOfSomSplits)
    # If the window size is the same as the slide size then there is no overlapping between splits
    splitTrainX = Preprocessing.SlidingWindowSplitter(trainX, windowSize, windowSize/slideDivisor)
    splitTestX = Preprocessing.SlidingWindowSplitter(testX, windowSize, windowSize/slideDivisor)
    numOfSplits = numOfSomSplits*slideDivisor-(slideDivisor-1)

    for gridSize in range(gridSizeStart, gridSizeEnd, 5):
        runName = RunNameHelper.GetRunName(method, somGridSize=gridSize, numOfSomSplits=numOfSplits)
        logFilePath = logFileDir + runName + ".txt"
        encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSplits, logFilePath=logFilePath,
                                             trainX=splitTrainX, testX=splitTestX, originalTrainSize=trainX.shape[0],
                                             originalTestSize=testX.shape[0])
        runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.MultipleSoms:
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfSomSplits, slideDivisor)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfSomSplits, slideDivisor)
    numOfSplits = numOfSomSplits*slideDivisor-(slideDivisor-1)

    for gridSize in range(gridSizeStart, gridSizeEnd, 5):
        runName = RunNameHelper.GetRunName(method=method, somGridSize=gridSize, numOfSomSplits=numOfSplits)
        logFilePath = logFileDir + runName + ".txt"

        for i in range(numOfSplits):
            nextTrainX, nextTestX = runSom(gridSize=gridSize, somSplit=1, logFilePath=logFilePath,
                                           trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i],
                                           originalTrainSize=trainX.shape[0], originalTestSize=testX.shape[0])
            if i == 0:
                newTrainX = nextTrainX
                newTestX = nextTestX
            else: # merge split projections
                newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
                newTestX = np.concatenate((newTestX, nextTestX), axis=1)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PCA:
    for i in range(pcaStart, pcaEnd):
        runName = RunNameHelper.GetRunName(method, numOfPcaComp=i)
        logFilePath = logFileDir + runName + ".txt"
        myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, i, logFilePath) # newTrainX = principal components
        newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PcaPlusEncoder:
    runName = RunNameHelper.GetRunName(method=method, autoEncoderType=aeType,
                                       numOfPcaComp=numOfPrinComp, numOfInputDim=numOfPrinComp)
    logFilePath = logFileDir + runName + ".txt"

    myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp, logFilePath)  # newTrainX = principal components
    newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

    newTrainX, newTestX = runAutoEncoder(autoEncoderType=aeType, runName=runName, logFilePath=logFilePath,
                                                   trainX=newTrainX, testX = newTestX)

    runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

elif method == Method.PcaPlusSom:
    gridSize = 40
    runName = RunNameHelper.GetRunName(method=method, somGridSize=gridSize,
                                       numOfSomSplits=numOfSomSplits, numOfPcaComp=numOfPrinComp)
    logFilePath = logFileDir + runName + ".txt"

    myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp, logFilePath)  # newTrainX = principal components
    newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

    splitTrainX = np.reshape(newTrainX, (newTrainX.shape[0] * numOfSomSplits, int(newTrainX.shape[1] / numOfSomSplits)))
    splitTestX = np.reshape(newTestX, (newTestX.shape[0] * numOfSomSplits, int(newTestX.shape[1] / numOfSomSplits)))

    encodedTrainX, encodedTestX = runSom(gridSize=gridSize, somSplit=numOfSomSplits, isCoordBased=isCoordBased,
                                         logFilePath=logFilePath, trainX=splitTrainX, testX=splitTestX,
                                         originalTrainSize=newTrainX.shape[0], originalTestSize=newTestX.shape[0])
    runKNN(encodedTrainX, encodedTestX, trainY, testY, logFilePath)

elif method == Method.MultipleEncodersAndSOMs:
    gridSize = 40
    splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfAeSplits, slideDivisor=1)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfAeSplits, slideDivisor=1)
    runName = RunNameHelper.GetRunName(method, autoEncoderType=aeType, somGridSize=gridSize,
                                       numOfSomSplits=numOfSomSplits, numOfAeSplits=numOfAeSplits,
                                       numOfInputDim=trainX.shape[1] / numOfAeSplits)
    logFilePath = logFileDir + runName + ".txt"

    for i in range(numOfAeSplits):
        nextTrainX, nextTestX = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                               trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i])
        if i == 0:
            newTrainX = nextTrainX
            newTestX = nextTestX
        else:  # merge split encodings
            newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
            newTestX = np.concatenate((newTestX, nextTestX), axis=1)

    splitByIndexTrainX = Preprocessing.SplitDataByIndex(newTrainX, numOfSomSplits, slideDivisor=1)
    splitByIndexTestX = Preprocessing.SplitDataByIndex(newTestX, numOfSomSplits, slideDivisor=1)
    for i in range(numOfSomSplits):
        nextTrainX, nextTestX = runSom(gridSize=gridSize, somSplit=1, logFilePath=logFilePath,
                                       trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i],
                                       originalTrainSize=trainX.shape[0], originalTestSize=testX.shape[0])
        if i == 0:
            newTrainX = nextTrainX
            newTestX = nextTestX
        else:  # merge split encodings
            newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
            newTestX = np.concatenate((newTestX, nextTestX), axis=1)

    runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

if sleepIndicator == 1:
    import os
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")