import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNNClassifier(object):

    # maxK - maximum number of k for kNN
    # cvNum - number of folds for cross validation
    def __init__(self, maxK, cvFolds, logFilePath):
        self.logFilePath = logFilePath

        paramGrid = {"n_neighbors": np.arange(1, maxK)} # All values for k we want to try
        # Use gridsearch to test all values for k, using 10-fold-cross validation
        self.knnGscv = GridSearchCV(KNeighborsClassifier(), paramGrid, cv=cvFolds, verbose=2)

    def train(self, trainX, trainY):
        tic = time.perf_counter()
        self.knnGscv.fit(trainX, trainY)
        toc = time.perf_counter()

        with open(self.logFilePath, "a") as resultsWriter:
            resultsWriter.write(f"KNN training time: {toc - tic:0.4f} seconds \r")

            resultsWriter.write("Best k: " + str(self.knnGscv.best_params_["n_neighbors"]) + "\r")
            resultsWriter.write(f"Best validation accuracy: {self.knnGscv.best_score_:0.4f} \r")

    def test(self, testX, testY):
        with open(self.logFilePath, "a") as resultsWriter:
            tic = time.perf_counter()
            resultsWriter.write(f"KNN testing accuracy: {self.knnGscv.score(testX, testY):0.4f} \r")
            toc = time.perf_counter()
            resultsWriter.write(f"KNN testing time: {toc - tic:0.4f} seconds \r")
