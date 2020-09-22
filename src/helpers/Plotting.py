from matplotlib import pyplot as plt
import numpy as np

# This class has been implemented for the sole purpose of visual aids in study documentation
from Preprocessing import Preprocessing


class Plotting:

    @staticmethod
    def PlotHeartbeats(allSamples, start, end, frequency):
        heartBeats = allSamples[start:end]
        heartBeatLength = heartBeats[0].size
        heartbeatDuration = heartBeatLength / frequency
        ts = np.linspace(0, heartbeatDuration, heartBeatLength)

        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(111)
        for i in range(end - start):
            ax.plot(ts, heartBeats[i])

        plt.grid()
        plt.show()

    @staticmethod
    def PlotSignal(signal, frequency):
        signalLength = signal.size
        signalDuration = signalLength / frequency
        ts = np.linspace(0, signalDuration, signalLength)

        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(111)
        ax.plot(ts, signal)
        plt.grid()
        plt.show()

    @staticmethod
    def PlotFourSubsections(subsections, frequency):
        subsectionLength = subsections[0].size
        subsectionDuration = subsectionLength / frequency
        ts = np.linspace(0, subsectionDuration, subsectionLength)
        fig = plt.figure(figsize=(14, 8))

        ax1 = fig.add_subplot(221)
        ax1.grid()
        plt.plot(ts, subsections[0])

        ax2 = fig.add_subplot(222, sharey=ax1)
        ax2.grid()
        ax2.plot(ts, subsections[1])

        ax3 = fig.add_subplot(223, sharey=ax1)
        ax3.grid()
        ax3.plot(ts, subsections[2])

        ax4 = fig.add_subplot(224, sharey=ax1)
        ax4.grid()
        ax4.plot(ts, subsections[3])

        plt.show()

samples = Preprocessing.LoadAllSamplesFromCsv("C:\\Users\\nenad\\PycharmProjects\\thesis\\data\\finalencoding-sample.csv", False)
#Plotting.PlotFourSubsections(samples, 500)
Plotting.PlotHeartbeats(samples, 0, 1, 500)