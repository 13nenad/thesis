import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import numpy as np

def plotMIDs(data, filePath):

    fig, ax = plt.subplots()
    cBarAx = fig.add_axes([0.8, 0.15, 0.03, 0.65]) # left, bottom, width, height
    fig.subplots_adjust(right=0.8)

    cmap = mpl.cm.binary
    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
    mpl.colorbar.ColorbarBase(cBarAx, cmap=cmap, norm=norm, orientation='vertical')

    ax.imshow(data, cmap=cmap)

    # draw gridlines
    ax.grid(axis='both', linestyle='-', color='k', linewidth=1)
    xAndY = np.arange(-0.5, data.shape[0], 1)
    ax.set_xticks(xAndY);
    ax.set_yticks(xAndY);

    xLabels = np.copy(xAndY) + 0.5
    xLabels = xLabels.astype(int)
    ax.set_xticklabels(xLabels)
    yLabels = np.copy(np.flip(xAndY)) + 0.5
    yLabels = yLabels.astype(int)
    ax.set_yticklabels(yLabels)

    plt.savefig(filePath)
    #plt.show()

def plotFittedLocations(locations, actuals, numOfClasses, gridSize, filePath, percentage):
    # Only plot the percentage of the data specified
    locations = locations[0: int(len(locations) * percentage / 100)]
    actuals = actuals[0: int(len(actuals) * percentage / 100)]

    fig, ax = plt.subplots()
    colors = ["red", "brown", "blue", "green", "orange", "purple", "cyan", "magenta", "black"]
    actuals -= 1 # Make the class list 0 based

    xData = []
    yData = []
    for i in range(numOfClasses):
        xData.append([])
        yData.append([])

    for i in range(len(locations)):
        actual = int(actuals[i])
        xData[actual].append(locations[i][0] + np.random.uniform(-0.45, 0.45, 1)[0])
        yData[actual].append(locations[i][1] + np.random.uniform(-0.45, 0.45, 1)[0])

    for i in range(numOfClasses):
        plt.scatter(xData[i], yData[i], 12, color=colors[i])

    # draw gridlines
    ax.grid(axis='both', linestyle='-', color='k', linewidth=1)
    xAndY = np.arange(-0.5, gridSize, 1)
    ax.set_xticks(xAndY);
    ax.set_yticks(xAndY);

    labels = np.copy(xAndY) + 0.5
    labels = labels.astype(int)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.savefig(filePath + "-Points" + str(len(locations)) + ".png")
    #plt.show()

#data = np.random.uniform(-0.4, 0.4, 1)
#actuals = np.array([1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 6.0])
#locations = [[0, 0], [1,1], [0, 0], [0, 0], [5, 2], [5, 5], [4, 3]]
#plotFittedLocations(np.array(locations), actuals, 9, 6, "", 100)

#plotMID(data)