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

class SampleType(enum.Enum):
   Raw = 1
   Filtered = 2

def GetNumOfSplits():
    print("Number of splits: ")
    return int(input())

def GetMethodIndex():
    print("1. Single Encoder\n2. Single SOM\n3. Encoder + SOM\n4. Multiple Encoders\n5. Multiple SOMs\n6. PCA Only"\
          "\n7. PCA + Encoder\n8. PCA + SOM")
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