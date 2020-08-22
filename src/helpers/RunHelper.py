import enum

class Method(enum.Enum):
   SingleEncoder = 1
   SingleSom = 2
   EncoderPlusSom = 3
   MultipleEncoders = 4
   MultipleSoms = 5
   PCA = 6
   PcaPlusEncoder = 7
   PcaPlusSom = 8
   MultipleEncodersAndSOMs = 9

def GetMethodIndex():
    print("1. Single Encoder\n2. Single SOM\n3. Encoder + SOM\n4. Multiple Encoders\n5. Multiple SOMs\n6. PCA"\
          "\n7. PCA + Encoder\n8. PCA + SOM\n9. Multiple Encoders + Multiple SOMs")
    return input()

def GetProjType():
    print("1. SOM projection is coordinate based\n2. SOM projection is label based")
    return input()

def GetAeType():
    print("Auto-encoder architecture (1-12): ")
    return int(input())

def GetGridSize():
    print("Grid size: ")
    return int(input())

def GetNumOfPrinComp():
    print("Number of principal components: ")
    return int(input())

def GetSlideDivisor():
    print("Slide size divisor (1 = no overlapping/sliding): ")
    return int(input())

def GetSleepIndicator():
    print("Put your device to sleep after run (1 = sleep, any other value = don't sleep): ")
    return int(input())

def GetNumOfAeSplits():
    print("Number of auto-encoder splits: ")
    return int(input())

def GetNumOfSomSplits():
    print("Number of SOM splits: ")
    return int(input())

def GetStartingGridSize():
    print("Starting SOM grid size: ")
    return int(input())

def GetEndingGridSize():
    print("Ending SOM grid size: ")
    return int(input()) + 1

def GetStartingAeType():
    print("Starting auto-encoder type (1-12): ")
    return int(input())

def GetEndingAeType():
    print("Ending auto-encoder type (1-12): ")
    return int(input()) + 1

def GetStartingPca():
    print("Starting principal components: ")
    return int(input())

def GetEndingPca():
    print("Ending principal components: ")
    return int(input()) + 1

def getAeArchStr(autoEncoderType, numOfInputDim):
    archStr = ""
    if autoEncoderType == 1: archStr = "AE-X-256-X"
    if autoEncoderType == 2: archStr = "AE-X-128-X"
    elif autoEncoderType == 3: archStr = "AE-X-64-X"
    elif autoEncoderType == 4: archStr = "AE-X-32-X"
    elif autoEncoderType == 5: archStr = "AE-X-16-X"
    elif autoEncoderType == 6: archStr = "AE-X-128-64-128-X"
    elif autoEncoderType == 7: archStr = "AE-X-256-128-256-X"
    elif autoEncoderType == 8: archStr = "AE-X-128-64-32-64-128-X"
    elif autoEncoderType == 9: archStr = "AE-X-256-128-64-128-256-X"
    elif autoEncoderType == 10: archStr = "AE-X-128-64-32-16-32-64-128-X"
    elif autoEncoderType == 11: archStr = "AE-X-256-128-64-32-64-128-256-X"
    elif autoEncoderType == 12: archStr = "AE-X-256-128-64-32-16-32-64-128-256-X"

    return archStr.replace("X", str(int(numOfInputDim)))

def getRunName(method, autoEncoderType=0, somGridSize=0, numOfSomSplits=0, numOfAeSplits=0,
               numOfPcaComp=0, numOfInputDim=0, slideDiv=0):

    if (method == Method.SingleEncoder or method == Method.EncoderPlusSom or method == Method.MultipleEncoders or
        method == Method.PcaPlusEncoder or method == Method.MultipleEncodersAndSOMs) and numOfInputDim == 0:
        raise ValueError("numOfInputDim needs to be specified if using an auto encoder.")

    runName = ""
    if method == Method.SingleEncoder:
        runName += getAeArchStr(autoEncoderType, numOfInputDim)
    elif method == Method.SingleSom:
        runName += "SOM-" + str(somGridSize)
    elif method == Method.EncoderPlusSom:
        runName += getAeArchStr(autoEncoderType, numOfInputDim)
        runName += "-SOM-" + str(somGridSize)
    elif method == Method.MultipleEncoders:
        runName += "Multiple-" + getAeArchStr(autoEncoderType, numOfInputDim)
        runName += "-Split-" + str(numOfAeSplits)
    elif method == Method.MultipleSoms:
        runName += "Multiple-SOM-" + str(somGridSize)
    elif method == Method.PCA:
        runName += "PCA-" + str(numOfPcaComp)
    elif method == Method.PcaPlusEncoder:
        aeType = getAeArchStr(autoEncoderType, numOfInputDim).replace("300", str(numOfPcaComp))
        runName += "PCA-" + str(numOfPcaComp) + "-" + aeType
    elif method == Method.PcaPlusSom:
        runName += "PCA-" + str(numOfPcaComp)
        runName += "-SOM-" + str(somGridSize)
    elif method == Method.MultipleEncodersAndSOMs:
        runName += "Multiple-" + getAeArchStr(autoEncoderType, numOfInputDim)
        runName += "-Split-" + str(numOfAeSplits)
        runName += "-Multiple-SOM-" + str(somGridSize)

    if numOfSomSplits != 0:
        runName += "-Split-" + str(numOfSomSplits)

    if slideDiv != 0:
        runName += "-SlideDiv-" + str(slideDiv)
        numOfOutputDim = numOfSomSplits * slideDiv - (slideDiv - 1)
        runName += "-TotalSplits-" + str(numOfOutputDim)

    return runName

def GetLogFilePath(logFileDir, method, aeType=0, gridSize=0, numOfSomSplits=0, numOfAeSplits=0,
                   numOfPcaComp=0, numOfInputDim=0, slideDiv=0, projType=""):
    runName = getRunName(method, autoEncoderType=aeType, somGridSize=gridSize,
                         numOfSomSplits=numOfSomSplits, numOfAeSplits=numOfAeSplits,
                         numOfPcaComp=numOfPcaComp, numOfInputDim=numOfInputDim, slideDiv=slideDiv)

    logFilePath = logFileDir + "\\" + runName + ".txt"

    printParams(logFilePath=logFilePath, method=method, aeType=aeType, numOfInputDim=numOfInputDim, gridSize=gridSize,
                numOfAeSplits=numOfAeSplits, numOfSomSplits=numOfSomSplits, slideDiv=slideDiv, projType=projType,
                numOfPcaComp=numOfPcaComp)

    return logFilePath


def printParams(logFilePath, method, projType="", aeType=0, numOfInputDim=0, gridSize=0, numOfPcaComp=0,
                slideDiv=0, numOfAeSplits=0, numOfSomSplits=0):
    if projType == "1": projType = "Coordinate-based"
    elif projType == "2": projType = "Label-based"

    with open(logFilePath, "a") as resultsWriter:
        resultsWriter.write(f"Method: {method} \r")
        if aeType != 0: resultsWriter.write(f"Auto-encoder architecture: {getAeArchStr(aeType, numOfInputDim)} \r")
        if numOfAeSplits != 0: resultsWriter.write(f"Number of auto-encoder splits: {numOfAeSplits} \r")
        if projType != "": resultsWriter.write(f"SOM projection: {projType} \r")
        if gridSize != 0: resultsWriter.write(f"Grid size: {gridSize} \r")
        if numOfSomSplits != 0: resultsWriter.write(f"Number of SOM splits: {numOfSomSplits} \r")
        if slideDiv != 0:
            resultsWriter.write(f"Slide size divisor: {slideDiv} \r")
            numOfOutputDim = numOfSomSplits * slideDiv - (slideDiv - 1)
            resultsWriter.write(f"Number of total SOM splits: {numOfOutputDim} \r")
        if numOfPcaComp != 0: resultsWriter.write(f"Number of PCA components: {numOfPcaComp} \r")
