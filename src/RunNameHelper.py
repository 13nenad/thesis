from RunHelper import Method

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

def GetRunName(method, autoEncoderType=0, somGridSize=0, numOfSomSplits=1, numOfAeSplits=1,
               numOfPcaComp=0, numOfInputDim=0):
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

    if numOfSomSplits != 1:
        runName += "-Split-" + str(numOfSomSplits)

    return runName