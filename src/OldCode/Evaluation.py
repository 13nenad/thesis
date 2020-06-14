import os
import random

import numpy as np

def getModelAccuracy(resultsTuple):
    TPs = resultsTuple[0]
    TNs = resultsTuple[1]
    FPs = resultsTuple[2]
    FNs = resultsTuple[3]

    return (TPs + TNs) / (TPs + TNs + FPs + FNs)

# Count the number of samples of each class that are in each square (coordinate)
def classCountInEachSquare(locations, actuals, numOfClasses, gridSize):
    classCntList = np.zeros((gridSize, gridSize, numOfClasses), dtype=int)

    for i in range(len(locations)):
        x = locations[i][0]
        y = locations[i][1]
        actual = actuals[i]
        classCntList[x][y][actual - 1] += 1

    return classCntList

# Calculate the percentage of samples of each class that are in each square
# E.g. calculate the percentage of class 1 samples that are in square [0, 0]
def calcClassPercInEachSquare(classCntList, actuals, numOfClasses, gridSize):
    actualTotals = np.zeros(9, dtype=int)
    for actual in actuals:
        actualTotals[actual - 1] += 1

    classPercList = np.zeros((gridSize, gridSize, numOfClasses))
    for x in range(gridSize):
        for y in range(gridSize):
            for c in range(numOfClasses):
                if classCntList[x][y][c] != 0:
                    classPercList[x][y][c] = classCntList[x][y][c] / actualTotals[c]

    return classPercList

# gridSize = 6
# locations = [[0, 0], [0, 2], [0, 1], [2, 2], [0, 1], [4, 5], [5, 4], [3, 2], [2, 3], [5, 5]]
# actuals = [1.0, 1.0, 5.0, 6.0, 4.0, 3.0, 2.0, 7.0, 8.0, 9.0]
# a = countClassNumInEachSquare(locations, actuals, 9, gridSize)
# b = calcClassPercInEachSquare(a, gridSize, 9, actuals)
# c = calcClassOfEachSquare(b, gridSize)

class Evaluation:

    # predictionsPerRegion - predictions for every region
    @staticmethod
    def SaveBestResults(predictionsPerRegion, actuals, regions, filePath, iteration):
        if len(predictionsPerRegion) != len(actuals):
            raise ValueError("Length of predictions doesn't match length of actuals")

        bestModelAccuracy, bestTP, bestTN, bestFP, bestFN, bestIndex = 0, 0, 0, 0, 0, 0
        for n in range(len(predictionsPerRegion)):
            TPs, TNs, FPs, FNs = 0, 0, 0, 0
            for i in range(len(predictionsPerRegion[n])):
                if predictionsPerRegion[n][i] == 1 and actuals[i] == 1:
                    TPs += 1
                elif predictionsPerRegion[n][i] == 1 and actuals[i] != 1:
                    FPs += 1
                elif predictionsPerRegion[n][i] == 0 and actuals[i] != 1:
                    TNs += 1
                elif predictionsPerRegion[n][i] == 0 and actuals[i] == 1:
                    FNs += 1

            modelAccuracy = getModelAccuracy((TPs, TNs, FPs, FNs))
            if modelAccuracy > bestModelAccuracy:
                bestModelAccuracy = modelAccuracy
                bestTP = TPs
                bestTN = TNs
                bestFP = FPs
                bestFN = FNs

        with open(filePath, "a") as resultsWriter:
            regionStr = str(regions[bestIndex])
            resultsWriter.write("Run:" + str(iteration) + "\r")
            resultsWriter.write("Region:" + regionStr[1:len(regionStr) - 1] + "\r")
            resultsWriter.write("TP:" + str(bestTP) + "\r")
            resultsWriter.write("TN:" + str(bestTN) + "\r")
            resultsWriter.write("FP:" + str(bestFP) + "\r")
            resultsWriter.write("FN:" + str(bestFN) + "\r")
            resultsWriter.write("Accuracy:" + str(modelAccuracy) + "\r")

        return bestModelAccuracy

    @staticmethod
    def SaveAvgAccuracy(accuracyScores, resultFilePath, numOfSamples):
        avgScore = 0
        for score in accuracyScores: avgScore += score
        avgScore /= 10
        with open(resultFilePath, "a") as resultsWriter:
            resultsWriter.write("Number of samples: " + str(numOfSamples) + "\r")
            resultsWriter.write("Average Accuracy:" + str(avgScore) + "\r\r")

    # TODO
    @staticmethod
    def LoadResultsTuple(filePath):
        resultsFile = open(filePath, 'r')
        results = resultsFile.readlines()
        TPs = int(results[0].split(":")[1])
        TNs = int(results[1].split(":")[1])
        FPs = int(results[2].split(":")[1])
        FNs = int(results[3].split(":")[1])

        return TPs, TNs, FPs, FNs

    # TODO
    @staticmethod
    def PickValidationWinner(directory):
        fileName, highestAccuracy = ("", 0)
        for fileName in os.listdir(directory):
            fileResults = Evaluation.LoadResultsTuple(directory + "/" + fileName)
            accuracy = Evaluation.GetModelAccuracy(fileResults)
            if accuracy > highestAccuracy[1]:
                highestAccuracy[1] = accuracy

    @staticmethod
    def CalcClassOfEachSquare(locations, actuals, numOfClasses, gridSize):
        classCountList = classCountInEachSquare(locations, actuals, numOfClasses, gridSize)
        classPercList = calcClassPercInEachSquare(classCountList, actuals, numOfClasses, gridSize)

        classListMethod1 = np.zeros((gridSize, gridSize), dtype=int)
        classListMethod2 = np.zeros((gridSize, gridSize), dtype=int)
        for x in range(gridSize):
            for y in range(gridSize):
                # Method 1- Get highest number of samples out of all classes in the square
                maxVal = np.amax(classCountList[x][y])
                if maxVal != 0: # Make sure there are some samples in the square
                    # Get an array of indices of the maximum value
                    maxIndices = np.where(classCountList[x][y] == maxVal)
                    # If more than one max found then randomly choose one. Also transform class back to one-based indexing
                    classListMethod1[x][y] = maxIndices[0][random.randint(0, len(maxIndices) - 1)] + 1

                # Method 2 - Get highest percentage number out of all classes
                maxVal = np.amax(classPercList[x][y])
                if maxVal != 0:  # Make sure there are some samples in the square
                    # Get an array of indices of the maximum value
                    maxIndices = np.where(classPercList[x][y] == maxVal)
                    # If more than one max found then randomly choose one. Also transform class back to one-based indexing
                    classListMethod2[x][y] = maxIndices[0][random.randint(0, len(maxIndices) - 1)] + 1

        # If a square contains 0 it means there are no samples in that square, and that square is not allocated a class
        return classListMethod1, classListMethod2

    @staticmethod
    def GetModelAccuracy(sampleLocs, actuals, classModel):
        correctCounter = 0
        for i in range(len(sampleLocs)):
            if classModel[sampleLocs[i][0], sampleLocs[i][1]] == actuals[i]:
                correctCounter+= 1

        return correctCounter / len(sampleLocs)