import time
import numpy as np
from sompy import SOMFactory

def translateToCoords(projections, gridSize):
    coordProjections = np.zeros((projections.shape[0], projections.shape[1]*2))
    for i in range(projections.shape[0]):
        for n in range(projections.shape[1]):
            coordProjections[i][n*2] = projections[i][n] % gridSize
            coordProjections[i][n*2+1] = int(projections[i][n] / gridSize)

    return coordProjections

class MySom(object):

    def __init__(self, trainX, gridSize, logFilePath):
        self.logFilePath = logFilePath

        self.som = SOMFactory().build(trainX, mapsize=[gridSize, gridSize], initialization='random', lattice="rect",
                                 normalization="None")

    def train(self, epochs):
        tic = time.perf_counter()
        self.som.train(train_rough_len=epochs, train_finetune_len=1)
        toc = time.perf_counter()

        with open(self.logFilePath, "a") as resultsWriter:
            resultsWriter.write(f"\rSOM training time: {toc - tic:0.4f} seconds \r")

        return toc - tic

    def project(self, dataX, isTrainData):
        tic = time.perf_counter()
        projectedDataX = self.som.project_data(dataX)
        toc = time.perf_counter()

        with open(self.logFilePath, "a") as resultsWriter:
            if isTrainData:
                resultsWriter.write(f"SOM training encoding time: {toc - tic:0.4f} seconds \r")
            else:
                resultsWriter.write(f"SOM testing encoding time: {toc - tic:0.4f} seconds \r")

        return projectedDataX, toc - tic
