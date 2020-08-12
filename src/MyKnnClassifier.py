import time
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class MyKnnClassifier(object):

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
            predY = self.knnGscv.predict(testX)

            mcm = multilabel_confusion_matrix(testY, predY)
            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]
            precList = tp / (tp + fp)
            recallList = tp / (tp + fn)
            specList = tn / (tn + fp)

            resultsWriter.write(f"KNN testing accuracy: {self.knnGscv.score(testX, testY):0.4f} \r")
            resultsWriter.write(f"KNN testing precision: {sum(precList) / len(precList):0.4f} \r")
            resultsWriter.write(f"KNN testing recall: {sum(recallList) / len(recallList):0.4f} \r")
            resultsWriter.write(f"KNN testing specificity: {sum(specList) / len(specList):0.4f} \r")
            toc = time.perf_counter()
            resultsWriter.write(f"KNN testing time: {toc - tic:0.4f} seconds \r")