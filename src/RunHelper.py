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
    print("1. Single Encoder\n2. Single SOM\n3. Encoder + SOM\n4. Multiple Encoders\n5. Multiple SOMs\n6. PCA Only"\
          "\n7. PCA + Encoder\n8. PCA + SOM\n9. Multiple Encoders + Multiple SOMs")
    return input()

def GetProjType():
    print("1. SOM projection is coordinate based\n2. SOM projection is label based")
    return input()

def GetAeType():
    print("Auto-encoder type (1-12): ")
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
    return int(input())

def GetStartingAeType():
    print("Starting auto-encoder type (1-12): ")
    return int(input())

def GetEndingAeType():
    print("Ending auto-encoder type (1-12): ")
    return int(input())

def GetStartingPca():
    print("Starting principal components: ")
    return int(input())

def GetEndingPca():
    print("Ending principal components: ")
    return int(input())