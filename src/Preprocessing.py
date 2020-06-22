from biosppy.signals import ecg
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os
from numpy import genfromtxt

class Preprocessing:

    @staticmethod
    def LoadIcbebDataFromMatlabFiles(directory):
        dataSignals = []
        for filename in os.listdir(directory):
            dataDict = loadmat(directory + "/" + filename)
            ecgNdArray = dataDict["ECG"]["data"]
            newRow = ecgNdArray[0][0][0]
            dataSignals.append(newRow)

        return dataSignals

    @staticmethod
    def LoadPhysioNetDataFromMatlabFiles(directory):
        dataSignals = []
        for filename in os.listdir(directory):
            if ".mat" in filename:
                dataDict = loadmat(directory + "/" + filename)
                dataSignals.append(dataDict["val"][0])

        return dataSignals

    @staticmethod
    def SaveAllSignalsToCsv(dataSignals, directory):
        for signal in dataSignals:
            np.savetxt(directory + '/Temp.csv', [signal], delimiter=",")
            nextSignal = open(directory + '/Temp.csv', 'r')
            lines = nextSignal.readlines()
            with open(directory + '/AllSignals.csv', "a") as allSignals:
                allSignals.write(lines[0])

    @staticmethod
    def SaveAllSamplesToCsv(samples, filePath):
        # Concatenate all the sample strings together before writing to a file
        sampleStr = ""
        counter = 0
        for i in range(len(samples)):
            line = np.array2string(samples[i], separator=',', max_line_width=999999)
            sampleStr += line[2:len(line)-1] + "\r"  # Remove brackets from list

            if len(samples) - i < 1000:
                if len(samples)-1 == i:
                    with open(filePath, "a") as allSignals:
                        allSignals.write(sampleStr) # Last write
            elif i % 1000 == 0:
                with open(filePath, "a") as allSignals:
                    allSignals.write(sampleStr)
                    sampleStr = ""

    # Return a list of numpy arrays i.e. list of ECG signals
    @staticmethod
    def LoadAllSignalsFromCsv(filePath):
        dataSignals = []
        allSignalsFile = open(filePath, 'r')
        signalLines = allSignalsFile.readlines()

        for signal in signalLines:
            signalData = np.fromstring(signal, sep=",")
            if len(signalData) > 2:
                dataSignals.append(signalData)

        return dataSignals

    # Return numpy array of numpy arrays i.e. array of ECG samples
    @staticmethod
    def LoadAllSamplesFromCsv(filePath, splitOutClasses, sampleLength):
        dataSamples = []
        allSamplesFile = open(filePath, 'r')
        sampleLines = allSamplesFile.readlines()

        for sample in sampleLines:
            sampleData = np.fromstring(sample, sep=",")
            if len(sampleData) > 2:
                if len(sampleData) != sampleLength:
                   raise ValueError("Length of sample is not " + sampleLength)
                dataSamples.append(sampleData)

        dataSamples = np.array(dataSamples)

        if splitOutClasses:
            dataY = dataSamples[:, 0].astype(dtype=int)
            dataX = np.delete(dataSamples, 0, 1)
            return dataX, dataY
        else:
            return dataSamples

    @staticmethod
    def GetRawSamplesFromSignals(dataSignals, sampleRate):
        allRawSamples = []
        for signal in dataSignals:
            label = signal[0]    # Get the label value
            signal = signal[1:]  # Getting rid of the label value
            rPeaks = ecg.christov_segmenter(signal=signal, sampling_rate=sampleRate)
            extractions = ecg.extract_heartbeats(signal=signal, rpeaks=rPeaks[0], sampling_rate=sampleRate,
                                                 before=0.2, after=0.4)
            heartbeats = extractions['templates']

            for heartbeat in heartbeats:
                heartbeat = np.insert(heartbeat, 0, label)
                allRawSamples.append(heartbeat)

        return allRawSamples

    @staticmethod
    def GetFilteredSamplesFromSignals(dataSignals):
        allFilteredSamples = []
        for signal in dataSignals:
            label = signal[0]    # Get the label value
            signal = signal[1:]  # Getting rid of the label value
            out = ecg.ecg(signal=signal, sampling_rate=500., show=False)
            heartbeats = out['templates']

            for heartbeat in heartbeats:
                heartbeat = np.insert(heartbeat, 0, label)
                allFilteredSamples.append(heartbeat)

        return allFilteredSamples

    @staticmethod
    def AddLabelsToSignals(signalData, labelFilePath, dataset):
        if dataset == 0: # ICBEB dataset
            labelData = genfromtxt(labelFilePath, delimiter=',', skip_header=1, usecols=(1,))

            for i in range(len(labelData)):
                signalData[i] = np.insert(signalData[i], 0, labelData[i])

        elif dataset == 1: # PhysioNet dataset
            labelData = genfromtxt(labelFilePath, delimiter=',', skip_header=0, usecols=(1,), dtype="U")
            filteredData = []

            # Convert labels to number and remove classifications which are not N, A or O
            for i in range(len(labelData)):
                if labelData[i] == "N":
                    filteredData.append(np.insert(signalData[i], 0, 0))
                elif labelData[i] == "A":
                    filteredData.append(np.insert(signalData[i], 0, 1))
                elif labelData[i] == "O":
                    filteredData.append(np.insert(signalData[i], 0, 2))

            signalData = filteredData

        return signalData

    @staticmethod
    def PlotFirstSampleOfNthSignal(allSamples, n):
        nthSignal = allSamples[n]
        firstHeartbeat = nthSignal[0]
        heartbeatDuration = firstHeartbeat.size / 500.
        ts = np.linspace(0, heartbeatDuration, firstHeartbeat.size)
        plt.plot(ts, firstHeartbeat)
        plt.grid()
        plt.show()

    # Split samples into a 80% training/validation set and a 20% testing set
    @staticmethod
    def SplitSamplesAndSave(samplesDir, originalFileName, trainingFileName, testingFileName, sampleLength):
        samples = Preprocessing.LoadAllSamplesFromCsv(samplesDir + originalFileName, False, sampleLength)
        np.random.shuffle(samples)  # shuffle all of the samples
        # Split samples into a 80%/20% split
        twoWaySplit = np.split(samples, [int(0.8 * len(samples))])
        Preprocessing.SaveAllSamplesToCsv(twoWaySplit[0], samplesDir + trainingFileName)
        Preprocessing.SaveAllSamplesToCsv(twoWaySplit[1], samplesDir + testingFileName)

    # Reduce data set into a randomised subset of specified n samples
    @staticmethod
    def ReduceSamples(samplesDir, originalFileName, newFileName, n):
        samples = Preprocessing.LoadAllSamplesFromCsv(samplesDir + originalFileName, False)
        np.random.shuffle(samples)  # shuffle all of the samples
        Preprocessing.SaveAllSamplesToCsv(samples[0: n], samplesDir + newFileName)

    @staticmethod
    def NormaliseData(trainX, testX):
        data = np.append(trainX, testX, 0)
        max = data.max()
        min = data.min()
        trainX = (trainX - min) / (max - min)
        testX = (testX - min) / (max - min)

        return trainX, testX

    @staticmethod
    def SplitDataByIndex(dataX, numOfSplits):
        # Example: (1000, 300) => (1000*somSplit, 300/somSplit) => (2000, 150)
        splitTrainX = np.reshape(dataX, (dataX.shape[0] * numOfSplits, int(dataX.shape[1] / numOfSplits)))

        splitByIndexTrainX = []
        for i in range(numOfSplits): # Gather all data of the same index together
            splitByIndexTrainX.append(splitTrainX[i::numOfSplits])

        return splitByIndexTrainX

    @staticmethod
    def AddRandomNoiseToSample(dataX):
        for i in range(len(dataX)):
            dataX[i] += np.random.normal(0, 1, dataX.shape[1]) # add random noise
        return dataX

#Preprocessing.ReduceSamples("C:/Dev/DataSets/ICBEB/RawSamples/", "TrainingAndValSet.csv", "SmallTrainingSet.csv", 50000)
#Preprocessing.ReduceSamples("C:/Dev/DataSets/ICBEB/RawSamples/", "TestingSet.csv", "SmallTestingSet.csv", 10000)

#signals = Preprocessing.LoadPhysioNetDataFromMatlabFiles("C:/Dev/DataSets/PhysioNet/MatlabData")
#Preprocessing.SaveAllSignalsToCsv(signals, "C:/Dev/DataSets/PhysioNet")

#allSignals = Preprocessing.LoadAllSignalsFromCsv("C:/Dev/DataSets/PhysioNet/AllSignals.csv")
#allSignals = Preprocessing.AddLabelsToSignals(allSignals, "C:/Dev/DataSets/PhysioNet/Labels.csv", 1)
#Preprocessing.SaveAllSignalsToCsv(allSignals, "C:/Dev/DataSets/PhysioNet")

#signal = Preprocessing.AddRandomNoiseToSample(allSignals[0])
#out = ecg.ecg(signal=signal, sampling_rate=300, show=True)

#rawSamples = Preprocessing.GetRawSamplesFromSignals(allSignals, 300)
#Preprocessing.SaveAllSamplesToCsv(rawSamples, "C:/Dev/DataSets/PhysioNet/Raw/AllSamples.csv")
#Preprocessing.SplitSamplesAndSave("C:/Dev/DataSets/PhysioNet/Raw/", "AllSamples.csv", "TrainingSet.csv", "TestingSet.csv", 181)

