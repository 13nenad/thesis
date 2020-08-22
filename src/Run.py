import numpy as np
from MyAutoEncoder import MyAutoEncoder
from MyKnnClassifier import MyKnnClassifier
from MyPCA import MyPCA
from Preprocessing import Preprocessing
from MySom import MySom, translateToCoords

from helpers.RunHelper import Method, GetMethodIndex, GetProjType, GetAeType, GetGridSize, GetNumOfPrinComp, \
    GetSlideDivisor, GetSleepIndicator, GetNumOfAeSplits, GetNumOfSomSplits, GetStartingGridSize, \
    GetEndingGridSize, GetStartingAeType, GetEndingAeType, GetStartingPca, GetEndingPca, GetLogFilePath, printParams


def runKNN(trainX, testX, trainY, testY, logFilePath):
    # Initialise, train and test KNN. Use 3-fold-cross validation. Grid search k values 1-9.
    knn = MyKnnClassifier(10, 3, logFilePath)
    knn.train(trainX, trainY)
    knn.test(testX, testY)

def runAutoEncoder(autoEncoderType, logFilePath, trainX, testX):
    # Initialise, train and test AutoEncoder
    autoEncoder = MyAutoEncoder(trainX.shape[1], autoEncoderType, logFilePath)
    trainingTime = autoEncoder.train(trainX, 256, 100)
    encodedTrainX, trainEncodingTime = autoEncoder.encode(trainX, True)  # This is what we pass to our SOM or KNN to train
    encodedTestX, testEncodingTime = autoEncoder.encode(testX, False)  # This is what we pass to out SOM or KNN to test

    return encodedTrainX, encodedTestX, trainingTime, trainEncodingTime, testEncodingTime

def runMultipleEncoder(numOfSplits, aeType, logFilePath, splitByIndexTrainX, splitByIndexTestX):
    totalTrainingTime = 0
    totalTrainEncodingTime = 0
    totalTestEncodingTime = 0

    for i in range(numOfSplits):
        nextTrainX, nextTestX, trainingTime, trainEncodingTime, testEncodingTime\
            = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                            trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i])

        totalTrainingTime += trainingTime
        totalTrainEncodingTime += trainEncodingTime
        totalTestEncodingTime += testEncodingTime

        if i == 0:
            newTrainX = nextTrainX
            newTestX = nextTestX
        else:  # merge split encodings
            newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
            newTestX = np.concatenate((newTestX, nextTestX), axis=1)

    with open(logFilePath, "a") as resultsWriter:
        resultsWriter.write(f"\rTotal encoder training time: {totalTrainingTime:0.4f} seconds \r")
        resultsWriter.write(f"Total encoder training encoding time: {totalTrainEncodingTime:0.4f} seconds \r")
        resultsWriter.write(f"Total encoder testing encoding time: {totalTestEncodingTime:0.4f} seconds \r")

    return newTrainX, newTestX

def runSom(gridSize, somSplit, logFilePath, trainX, testX, originalTrainSize, originalTestSize,
           isCoordBased=True, isMultipleSom=False):
    mySom = MySom(trainX, gridSize, logFilePath)
    trainingTime = mySom.train(100)

    projectedTrainX, trainEncodingTime = mySom.project(trainX, isTrainData=True)
    projectedTestX, testEncodingTime = mySom.project(testX, isTrainData=False)

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
        return projectedTrainX, projectedTestX, trainingTime, trainEncodingTime, testEncodingTime

    return encodedTrainX, encodedTestX, trainingTime, trainEncodingTime, testEncodingTime

def runMultipleSom(numOfSplits, gridSize, logFilePath, splitByIndexTrainX, splitByIndexTestX, trainX, testX):
    totalTrainingTime = 0
    totalTrainEncodingTime = 0
    totalTestEncodingTime = 0

    for i in range(numOfSplits):
        nextTrainX, nextTestX, trainingTime, trainEncodingTime, testEncodingTime\
            = runSom(gridSize=gridSize, somSplit=1, logFilePath=logFilePath,
                    trainX=splitByIndexTrainX[i], testX=splitByIndexTestX[i],
                    originalTrainSize=trainX.shape[0], originalTestSize=testX.shape[0])

        totalTrainingTime += trainingTime
        totalTrainEncodingTime += trainEncodingTime
        totalTestEncodingTime += testEncodingTime

        if i == 0:
            newTrainX = nextTrainX
            newTestX = nextTestX
        else:  # merge split projections
            newTrainX = np.concatenate((newTrainX, nextTrainX), axis=1)
            newTestX = np.concatenate((newTestX, nextTestX), axis=1)

    with open(logFilePath, "a") as resultsWriter:
        resultsWriter.write(f"\rTotal SOM training time: {totalTrainingTime:0.4f} seconds \r")
        resultsWriter.write(f"Total SOM training encoding time: {totalTrainEncodingTime:0.4f} seconds \r")
        resultsWriter.write(f"Total SOM testing encoding time: {totalTestEncodingTime:0.4f} seconds \r")

    return newTrainX, newTestX

def main():
    import os
    baseDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "data"))
    logFileDir = os.path.abspath(os.path.join(baseDir, '..', "results"))

    runVal = GetMethodIndex()
    if runVal == "1":
        method = Method.SingleEncoder
        aeTypeStart = GetStartingAeType()
        aeTypeEnd = GetEndingAeType()
    elif runVal == "2":
        method = Method.SingleSom
        numOfSomSplits = GetNumOfSomSplits()
        slideDiv = GetSlideDivisor()
        projType = GetProjType()
        gridSizeStart = GetStartingGridSize()
        gridSizeEnd = GetEndingGridSize()
        if projType == "1": isCoordBased = True
        else: isCoordBased = False
    elif runVal == "3":
        method = Method.EncoderPlusSom
        aeType = GetAeType()
        gridSize = GetGridSize()
        numOfSomSplits = GetNumOfSomSplits()
        projType = GetProjType()
        if projType == "1": isCoordBased = True
        else: isCoordBased = False
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
        gridSize = GetGridSize()
        projType = GetProjType()
        if projType == "1": isCoordBased = True
        else: isCoordBased = False
        numOfSomSplits = GetNumOfSomSplits()
    elif runVal == "9":
        method = Method.MultipleEncodersAndSOMs
        aeType = GetAeType()
        numOfAeSplits = GetNumOfAeSplits()
        gridSize = GetGridSize()
        projType = GetProjType()
        numOfSomSplits = GetNumOfSomSplits()
        slideDiv = GetSlideDivisor()

    sleepIndicator = GetSleepIndicator()
    # Initialise training, validation and testing data
    trainX, trainY = Preprocessing.LoadAllSamplesFromCsv(os.path.join(baseDir, "TrainingSet.csv"), True)
    testX, testY = Preprocessing.LoadAllSamplesFromCsv(os.path.join(baseDir, "TestingSet.csv"), True)

### Used for testing purposes ####
    #trainX = trainX[0:1000]
    #trainY = trainY[0:1000]
    #testX = testX[0:1000]
    #testY = testY[0:1000]
##################################

    if method == Method.SingleEncoder:
        for aeType in range(aeTypeStart, aeTypeEnd):
            logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, aeType=aeType,
                                         numOfInputDim=trainX.shape[1])

            aeResult = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                                 trainX=trainX, testX=testX)

            runKNN(aeResult[0], aeResult[1], trainY, testY, logFilePath)

    elif method == Method.SingleSom:
        windowSize = int(trainX.shape[1] / numOfSomSplits)
        splitTrainX = Preprocessing.SlidingWindowSplitter(dataX=trainX, windowSize=windowSize,
                                                          slide=windowSize / slideDiv)
        splitTestX = Preprocessing.SlidingWindowSplitter(dataX=testX, windowSize=windowSize,
                                                         slide=windowSize / slideDiv)
        numOfOutputDim = numOfSomSplits * slideDiv - (slideDiv - 1)

        for gridSize in range(gridSizeStart, gridSizeEnd, 5):
            logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, gridSize=gridSize,
                                         numOfSomSplits=numOfSomSplits, slideDiv=slideDiv, projType=projType)

            somResult = runSom(gridSize=gridSize, somSplit=numOfOutputDim, logFilePath=logFilePath,
                                                 trainX=splitTrainX, testX=splitTestX,
                                                 originalTrainSize=trainX.shape[0],
                                                 originalTestSize=testX.shape[0])

            runKNN(somResult[0], somResult[1], trainY, testY, logFilePath)

    elif method == Method.EncoderPlusSom:
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, aeType=aeType, gridSize=gridSize,
                                     numOfSomSplits=numOfSomSplits, numOfInputDim=trainX.shape[1], projType=projType)

        aeResult = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath, trainX=trainX, testX=testX)

        windowSize = int(aeResult[0].shape[1] / numOfSomSplits)
        splitTrainX = Preprocessing.SlidingWindowSplitter(dataX=aeResult[0], windowSize=windowSize, slide=windowSize)
        splitTestX = Preprocessing.SlidingWindowSplitter(dataX=aeResult[1], windowSize=windowSize, slide=windowSize)

        somResult = runSom(gridSize=gridSize, somSplit=numOfSomSplits, logFilePath=logFilePath,
                                             trainX=splitTrainX, testX=splitTestX,
                                             originalTrainSize=aeResult[0].shape[0],
                                             originalTestSize=aeResult[1].shape[0])

        runKNN(somResult[0], somResult[1], trainY, testY, logFilePath)

    elif method == Method.MultipleEncoders:
        splitByIndexTrainX = Preprocessing.SplitDataByIndex(dataX=trainX, numOfSplits=numOfAeSplits, slideDivisor=1)
        splitByIndexTestX = Preprocessing.SplitDataByIndex(dataX=testX, numOfSplits=numOfAeSplits, slideDivisor=1)

        for aeType in range(aeTypeStart, aeTypeEnd):
            logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, aeType=aeType,
                                         numOfAeSplits=numOfAeSplits, numOfInputDim=trainX.shape[1] / numOfAeSplits)

            newTrainX, newTestX = runMultipleEncoder(numOfAeSplits, aeType, logFilePath,
                                                     splitByIndexTrainX, splitByIndexTestX)

            runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

    elif method == Method.MultipleSoms:
        splitByIndexTrainX = Preprocessing.SplitDataByIndex(dataX=trainX, numOfSplits=numOfSomSplits,
                                                            slideDivisor=slideDiv)
        splitByIndexTestX = Preprocessing.SplitDataByIndex(dataX=testX, numOfSplits=numOfSomSplits,
                                                           slideDivisor=slideDiv)
        numOfOutputDim = numOfSomSplits * slideDiv - (slideDiv - 1)

        for gridSize in range(gridSizeStart, gridSizeEnd, 5):
            logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, gridSize=gridSize,
                                         numOfSomSplits=numOfSomSplits, slideDiv=slideDiv)

            newTrainX, newTestX = runMultipleSom(numOfOutputDim, gridSize, logFilePath, splitByIndexTrainX,
                                                 splitByIndexTestX, trainX, testX)

            runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

    elif method == Method.PCA:
        for i in range(pcaStart, pcaEnd):
            logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, numOfPcaComp=i)

            myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, i, logFilePath)  # newTrainX = principal components
            newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

            runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

    elif method == Method.PcaPlusEncoder:
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, aeType=aeType,
                                     numOfPcaComp=numOfPrinComp, numOfInputDim=numOfPrinComp)

        myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp,
                                                    logFilePath)  # newTrainX = principal components
        newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

        aeResult = runAutoEncoder(autoEncoderType=aeType, logFilePath=logFilePath,
                                             trainX=newTrainX, testX=newTestX)

        runKNN(aeResult[0], aeResult[1], trainY, testY, logFilePath)

    elif method == Method.PcaPlusSom:
        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, gridSize=gridSize,
                                     numOfSomSplits=numOfSomSplits, numOfPcaComp=numOfPrinComp, projType=projType)

        myPca, newTrainX = MyPCA.ReduceTrainingData(trainX, numOfPrinComp,
                                                    logFilePath)  # newTrainX = principal components
        newTestX = MyPCA.ReduceTestingData(testX, myPca, logFilePath)

        windowSize = int(newTrainX.shape[1] / numOfSomSplits)
        splitTrainX = Preprocessing.SlidingWindowSplitter(dataX=newTrainX, windowSize=windowSize, slide=windowSize)
        splitTestX = Preprocessing.SlidingWindowSplitter(dataX=newTestX, windowSize=windowSize, slide=windowSize)

        somResult = runSom(gridSize=gridSize, somSplit=numOfSomSplits, isCoordBased=isCoordBased,
                                             logFilePath=logFilePath, trainX=splitTrainX, testX=splitTestX,
                                             originalTrainSize=newTrainX.shape[0], originalTestSize=newTestX.shape[0])

        runKNN(somResult[0], somResult[1], trainY, testY, logFilePath)

    elif method == Method.MultipleEncodersAndSOMs:
        splitByIndexTrainX = Preprocessing.SplitDataByIndex(trainX, numOfAeSplits, slideDivisor=1)
        splitByIndexTestX = Preprocessing.SplitDataByIndex(testX, numOfAeSplits, slideDivisor=1)

        logFilePath = GetLogFilePath(method=method, logFileDir=logFileDir, aeType=aeType, gridSize=gridSize,
                                     numOfSomSplits=numOfSomSplits, slideDiv=slideDiv, numOfAeSplits=numOfAeSplits,
                                     numOfInputDim=trainX.shape[1] / numOfAeSplits, projType=projType)

        newTrainX, newTestX = runMultipleEncoder(numOfAeSplits, aeType, logFilePath,
                                                 splitByIndexTrainX, splitByIndexTestX)

        splitByIndexTrainX = Preprocessing.SplitDataByIndex(newTrainX, numOfSomSplits, slideDivisor=slideDiv)
        splitByIndexTestX = Preprocessing.SplitDataByIndex(newTestX, numOfSomSplits, slideDivisor=slideDiv)

        newTrainX, newTestX = runMultipleSom(numOfSomSplits, gridSize, logFilePath, splitByIndexTrainX,
                                             splitByIndexTestX, newTrainX, newTestX)

        runKNN(newTrainX, newTestX, trainY, testY, logFilePath)

    if sleepIndicator == 1:
        import os
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

main()