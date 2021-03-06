#### Instructions
1. Using your python IDE pull down this repository
2. Insert your training set called "TrainingSet.csv" and your testing set called "TestingSet.csv" into the "data" folder.
3. Run the Run.py file and follow the prompts. For the first prompt enter the digit which corresponds to the method number.
4. After completion, it will output a txt file into the "results" folder. The name of the file should give an idea about which method and the parameters used.

Note: The last prompt will ask whether you want your device to be put to sleep after your method is run. Enter 1 if you do, any other character if you don't.

#### Output File
The output file will contain:
* The parameters you have chosen. These will always be at the top.
* The training time of your method chosen. 
* The training encoding time i.e. the time it took to encode all of the training samples.
* The testing encoding time i.e. the time it took to encode all of the testing samples.
* The kNN training time
* The best value for k (always between 1-9, inclusive)
* The associated best validation accuracy
* The evaluation metrics; accuracy, precision, recall and specificity
* The kNN classification time.

Note: If the method chosen is of multiple nature, it will in addition display the totals of each of the training and encoding times. 

#### Extras
Since the hexagonal topology and bubble neighbourhood function were only used for one experiment each I did not bother 
to make them configurable. If you would like to try these parameters you can go to line 19 in MySom.py and:
* Replace lattice="rect" with lattice="hexa"
* Insert neighborhood="bubble" as an argument (default is gaussian)

#### Example Input/Output
1. Single Encoder
2. Single SOM
3. Encoder + SOM
4. Multiple Encoders
5. Multiple SOMs
6. PCA Only
7. PCA + Encoder
8. PCA + SOM
9. Multiple Encoders + Multiple SOMs
10. No Dimensionality Reduction

->> 2 (Indicates you want to use the Single SOM method)

Number of SOM splits:

->> 10

Slide size divisor (1 = no overlapping/sliding):

->> 1

Starting SOM grid size:

->> 5

Ending SOM grid size:

->> 50 (This will now train and test on grid sizes 5,10,15,20,25,30,35,40,45,50)

Put your device to sleep after run (1 = sleep, any other value = don't sleep):

->> 2

#### Definitions
numOfSomSplits, numOfAeSplits - The number of chunks you want to split each sample into. Each piece will be projected/encoded and then all of the encodings from the same original sample will be concatenated to make a master encoding.

slideDiv - Slide Divisor, is used to control how many points you want to slide, creating a sliding
           window. sliding window = sampleDim/numOfSplits/slideDiv
           
coordBasedVal - Only used for SOMs. It indicates whether the output of the SOM will be a
                2 dimensional coordinate or just the index of the square. E.g. 3x3 = 9 squares
                
aeTypeStart, aeTypeEnd - Auto-encoder architecture type, specified in MyAutoEncoder.py

saveModel - The single and multiple autoencoder methods have the option of saving the trained model in the models folder
            and this model can then be loaded and reused.

The program currently has 9 different methods that it can run:
1. A single auto-encoder. It iterates over pre-defined architectures. You need to specify:
           aeTypeStart
           aeTypeEnd
           saveModel

2. A single SOM. It iterates over a specified number of different grid sizes, step size = 5. You will need to specify:
           
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
       aeTypeEnd
       saveModel

5. Multiple SOMs. It iterates over a specified number of different grid sizes, step size = 5. You will need to specify:
    
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

9. Multiple auto-encoders + multiple SOMs. You will need to specify:
    
       aeType
       slideDivisor
       numOfAeSplits
       gridSize
       numOfSomSplits
       slideDiv
       
10. No Dimensionality Reduction. You do not need to specify any further arguments.
    This will purely only run the kNN classifier.
    
11. Load Encoder. You do not need to specify any further arguments.
    This will load the encoder model present in the models folder.
    If there are no model files present or if more than one model files in this location, an exception is raised.
    
12. Load Multiple Encoders. You do not need to specify any further arguments.
    This will load all of the models present in the models folder.
    The user must ensure that all of the models are associated each other.
    The Multiple Encoders method will launch and it will split the samples into however many model files are present.
    


