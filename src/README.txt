Definitions:
numOfSomSplits, numOfAeSplits - The number of chunks you want to split each sample into
slideDivisor - is used to control how many points you want to slide, creating a sliding
               window. sliding window = sampleSize/numOfSplits/slideDivisor
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
You must specify the directory string on line 63 of where these 2 files live.
You must also have a folder called "KNN Results" inside this directory.
Then you just run the Run.py file and follow the prompts. It will output a txt file in the KNN Results folder.

Note: The last prompt will ask whether you want your device to be put to sleep after your method is run.
      Enter 1 if you do, anything else if you don't.



