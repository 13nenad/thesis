from RunHelper import Method

def getAeType(autoEncoderType):
    if autoEncoderType == 1: return "AE-300-256-300"
    if autoEncoderType == 2: return "AE-300-128-300"
    elif autoEncoderType == 3: return "AE-300-64-300"
    elif autoEncoderType == 4: return "AE-300-32-300"
    elif autoEncoderType == 5: return "AE-300-16-300"
    elif autoEncoderType == 6: return "AE-300-128-64-128-300"
    elif autoEncoderType == 7: return "AE-300-256-128-256-300"
    elif autoEncoderType == 8: return "AE-300-128-64-32-64-128-300"
    elif autoEncoderType == 9: return "AE-300-256-128-64-128-256-300"
    elif autoEncoderType == 10: return "AE-300-128-64-32-16-32-64-128-300"
    elif autoEncoderType == 11: return "AE-300-256-128-64-32-64-128-256-300"
    elif autoEncoderType == 12: return "AE-300-256-128-64-32-16-32-64-128-256-300"

def GetRunName(method, autoEncoderType=0, somGridSize=0, numOfSplits=1, normalise=False, numOfPcaComp=0):
    runName = ""
    if method == Method.SingleEncoder:
        runName += getAeType(autoEncoderType)
    elif method == Method.SingleSom:
        runName += "SOM-" + str(somGridSize)
    elif method == Method.EncoderPlusSom:
        runName += getAeType(autoEncoderType)
        runName += "-SOM-" + str(somGridSize)
    elif method == Method.MultipleEncoders:
        runName += "Multiple-" + getAeType(autoEncoderType)
    elif method == Method.MultipleSoms:
        runName += "Multiple-SOM-" + str(somGridSize)
    elif method == Method.PCA:
        runName += "PCA-" + str(numOfPcaComp)
    elif method == Method.PcaPlusEncoder:
        aeType = getAeType(autoEncoderType).replace("300", str(numOfPcaComp))
        runName += "PCA-" + str(numOfPcaComp) + "-" + aeType
    elif method == Method.PcaPlusSom:
        runName += "PCA-" + str(numOfPcaComp)
        runName += "-SOM-" + str(somGridSize)

    if numOfSplits != 1:
        runName += "-Split-" + str(numOfSplits)

    if normalise:
        runName += "-Norm"

    return runName