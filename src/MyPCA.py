import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class MyPCA(object):

    @staticmethod
    def ReduceTrainingData(trainX, numOfComp, logFilePath):
        tic = time.perf_counter()
        pca = PCA(n_components=numOfComp)
        principalComponents = pca.fit_transform(trainX)
        toc = time.perf_counter()

        with open(logFilePath, "a") as resultsWriter:
            resultsWriter.write(f"PCA training samples time: {toc - tic:0.4f} seconds \r")

        return pca, principalComponents

    @staticmethod
    def ReduceTestingData(testX, pca, logFilePath):
        tic = time.perf_counter()
        principalComponents = pca.transform(testX)
        toc = time.perf_counter()

        with open(logFilePath, "a") as resultsWriter:
            resultsWriter.write(f"PCA testing samples time: {toc - tic:0.4f} seconds \r")

        return principalComponents

    @staticmethod
    def variancePlot(pca):
        # Plot the explained variances
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('Variance %')
        plt.xticks(features)
        plt.show()