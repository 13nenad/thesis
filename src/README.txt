Definitions:
numOfSomSplits, numOfAeSplits - The number of chunks you want to split each sample into
slideDiv - Slide Divisor, is used to control how many points you want to slide, creating a sliding
           window. sliding window = sampleDim/numOfSplits/slideDiv
coordBasedVal - Only used for SOMs. It indicates whether the output of the SOM will be a
                2 dimensional coordinate or just the index of the square. E.g. 3x3 = 9 squares
aeTypeStart, aeTypeEnd - Auto-encoder architecture type, specified in Autoencoder.py

The program currently has 9 different methods that it can run:
1. A single auto-encoder. It loops over various pre-defined architectures. You need to specify:
    aeTypeStart
    aeTypeEnd

2. A single SOM. It loops over a specified number of different grid sizes, step size = 5. You will need to specify:
    numOfSplits
    slideDivisor
    coordBasedVal
    gridSizeStart
    gridSizeEnd

3. An auto-encoder + SOM. You will need to specify:
    aeType - Auto-encoder architecture type (1-13)
    gridSize
    numOfSomSplits
    coordBasedVal

4. Multiple auto-encoders. You will need to specify:
    numOfAeSplits
    aeTypeStart
    aeTypeEn

5. Multiple SOMs. It loops over a specified number of different grid sizes, step size = 5. You will need to specify:
    numOfSomSplits
    slideDivisor
    gridSizeStart
    gridSizeEnd

6. PCA. Loops over a specified number of principal components. You will need to specify:
    pcaStart
    pcaEnd

7. PCA + auto-encoder. You will need to specify:
    numOfPrincComp
    aeType

8. PCA + SOM. You will need to specify:
    numOfPrincComp
    numOfSomSplits

9. Multiple auto-encoders + multiple SOMs. The grid size is fixed to 40. You will need to specify:
    aeType
    slideDivisor
    numOfAeSplits
    numOfSomSplits

Instructions:
You must have a training set called "TrainingSet.csv" and a testing set called "TestingSet.csv".
You must specify the directory string on line 65 of where these 2 files live.
You must also have a folder called "KNN Results" inside this same directory.
Then you just run the Run.py file and follow the prompts.
For the first prompt enter the digit which corresponds to the method number.
After completion, it will output a txt file in the KNN Results folder. The name of the file should give an idea
about which method and the parameters used.

Note: The last prompt will ask whether you want your device to be put to sleep after your method is run.
      Enter 1 if you do, any other character if you don't.

Example Input:
1. Single Encoder
2. Single SOM
3. Encoder + SOM
4. Multiple Encoders
5. Multiple SOMs
6. PCA Only
7. PCA + Encoder
8. PCA + SOM
9. Multiple Encoders + Multiple SOMs
>> 2 (Indicates I want to use the Single SOM method)
Number of SOM splits:
>> 10 (Indicates I want to split each sample into 10 pieces - each piece will be projected as a coordinate and the
        projections coming from the same original sample will be concatenated to make an encoding.
Slide size divisor (1 = no overlapping/sliding):
>> 1
Starting SOM grid size:
>> 5
Ending SOM grid size:
>> 50 (This will now train and test on grid sizes 5,10,15,20,25,30,35,40,45,50)
Put your device to sleep after run (1 = sleep, any other value = don't sleep):
>> 2

