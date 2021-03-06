import time

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow_core.python.keras import losses, Sequential

class MyAutoEncoder(object):
    # archType - 1 => 300|256|300 : archType - 2 => 300|128|300
    # archType - 3 => 300|64|300  : archType - 4 => 300|32|300
    # archType - 5 => 300|16|300  : archType - 6 => 300|128|64|128|300
    # archType - 7 => 300|256|128|128|256|300           : archType - 8 => 300|128|64|32|64|128|300
    # archType - 9 => 300||256|128|64|128|256|300       : archType - 10 => 300|128|64|32|16|32|64|128|300
    # archType - 11 => 300|256|128|64|32|64|128|256|300 : archType - 12 => 300|256|128|64|32|16|32|64|128|256|300
    def __init__(self, logFilePath, inputDim=0, archType=0):
        self.logFilePath = logFilePath
        if archType == 0: return # We are loading a saved model

        # Create auto encoder+decoder
        self.autoEncoderModel = Sequential()
        self.autoEncoderModel.add(Dense(inputDim, input_shape=(inputDim,), activation='relu')) # Input layer

        if archType == 1:
            self.autoEncoderModel.add(Dense(256, activation='relu'))
        elif archType == 2:
            self.autoEncoderModel.add(Dense(128, activation='relu'))
        elif archType == 3:
            self.autoEncoderModel.add(Dense(64, activation='relu'))
        elif archType == 4:
            self.autoEncoderModel.add(Dense(32, activation='relu'))
        elif archType == 5:
            self.autoEncoderModel.add(Dense(16, activation='relu'))
        elif archType == 6:
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
        elif archType == 7:
            self.autoEncoderModel.add(Dense(256, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(256, activation='relu'))
        elif archType == 8:
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(32, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
        elif archType == 9:
            self.autoEncoderModel.add(Dense(256, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(256, activation='relu'))
        elif archType == 10:
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(32, activation='relu'))
            self.autoEncoderModel.add(Dense(16, activation='relu'))
            self.autoEncoderModel.add(Dense(32, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
        elif archType == 11:
            self.autoEncoderModel.add(Dense(256, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(32, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(256, activation='relu'))
        elif archType == 12:
            self.autoEncoderModel.add(Dense(256, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(32, activation='relu'))
            self.autoEncoderModel.add(Dense(16, activation='relu'))
            self.autoEncoderModel.add(Dense(32, activation='relu'))
            self.autoEncoderModel.add(Dense(64, activation='relu'))
            self.autoEncoderModel.add(Dense(128, activation='relu'))
            self.autoEncoderModel.add(Dense(256, activation='relu'))
        else:
            raise ValueError("Incorrect architecture type given.")

        self.autoEncoderModel.add(Dense(inputDim, activation='relu')) # Output layer
        self.autoEncoderModel.compile(optimizer='adam', loss=losses.MSE)
        self.autoEncoderModel.summary()

        # Create encoder
        inputSample = Input(shape=(inputDim,))
        inputLayer = self.autoEncoderModel.layers[0]
        if 0 < archType < 6:
            layerTwo = self.autoEncoderModel.layers[1]
            self.encoderModel = Model(inputSample, layerTwo(inputLayer(inputSample)))
        elif archType < 8:
            layerTwo = self.autoEncoderModel.layers[1]
            layerThree = self.autoEncoderModel.layers[2]
            self.encoderModel = Model(inputSample, layerThree(layerTwo(inputLayer(inputSample))))
        elif archType < 10:
            layerTwo = self.autoEncoderModel.layers[1]
            layerThree = self.autoEncoderModel.layers[2]
            layerFour = self.autoEncoderModel.layers[3]
            self.encoderModel = Model(inputSample, layerFour(layerThree(layerTwo(inputLayer(inputSample)))))
        elif archType < 12:
            layerTwo = self.autoEncoderModel.layers[1]
            layerThree = self.autoEncoderModel.layers[2]
            layerFour = self.autoEncoderModel.layers[3]
            layerFive = self.autoEncoderModel.layers[4]
            self.encoderModel = Model(inputSample, layerFive(layerFour(layerThree(layerTwo(inputLayer(inputSample))))))
        elif archType == 12:
            layerTwo = self.autoEncoderModel.layers[1]
            layerThree = self.autoEncoderModel.layers[2]
            layerFour = self.autoEncoderModel.layers[3]
            layerFive = self.autoEncoderModel.layers[4]
            layerSix = self.autoEncoderModel.layers[5]
            self.encoderModel = Model(inputSample,
                                      layerSix(layerFive(layerFour(layerThree(layerTwo(inputLayer(inputSample)))))))

        self.encoderModel.summary()

    def train(self, trainX, batchSize, epochs, isDenoising=False):
        tic = time.perf_counter()
        inputLayer = trainX
        if isDenoising: # add some noise to the input layer
            inputLayer = trainX + np.random.normal(0, 1, trainX.shape) / 2

        self.autoEncoderModel.fit(inputLayer, trainX, epochs=epochs, batch_size=batchSize,
                                  shuffle=True, validation_split=0.2)
        toc = time.perf_counter()

        with open(self.logFilePath, "a") as resultsWriter:
            resultsWriter.write(f"AutoEncoder training time: {toc - tic:0.4f} seconds \r")

        return toc - tic

    def encode(self, dataX, isTrainData):
        tic = time.perf_counter()
        encodedDataX = self.encoderModel.predict(dataX)
        toc = time.perf_counter()
        if isTrainData:
            with open(self.logFilePath, "a") as resultsWriter:
                resultsWriter.write(f"AutoEncoder training encoding time: {toc - tic:0.4f} seconds \r")
        else:
            with open(self.logFilePath, "a") as resultsWriter:
                resultsWriter.write(f"AutoEncoder testing encoding time: {toc - tic:0.4f} seconds \r\r")

        return encodedDataX, toc - tic