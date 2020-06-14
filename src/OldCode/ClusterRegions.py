
def getLeftOverRegion(regionA, gridSize):
    regionB = []
    for x in range(gridSize):
        for y in range(gridSize):
            coord = int(str(x) + str(y))
            region = convert2dTo1d(regionA)
            if coord not in region:
                regionB.append([x, y])

    return regionB

def convert2dTo1d(twoD):
    oneD = []
    for coord in twoD:
        oneD.append(int(str(coord[0]) + str(coord[1])))

    return oneD

def convert1dTo2d(oneD):
    twoD = []
    for coord in oneD:
        if len(str(coord)) < 2:
            twoD.append([0, int(str(coord)[0])])
        else:
            twoD.append([int(str(coord)[0]), int(str(coord)[1])])

    return twoD

def convert3dTo2d(threeD):
    twoD = []
    for x in range(len(threeD)):
        oneD = []
        for y in range(len(threeD[x])):
            oneD.append(int(str(threeD[x][y][0]) + str(threeD[x][y][1])))

        twoD.append(oneD)

    return twoD

def convert2dTo3d(twoD):
    threeD = []
    for x in range(len(twoD)):
        new2D = []
        for y in range(len(twoD[x])):
            v1 = twoD[x][y]
            if v1 < 10:
                new2D.append([0, v1])
            else:
                new2D.append([int(str(twoD[x][y])[0]), int(str(twoD[x][y])[1])])

        threeD.append(new2D)

    return threeD

def removeDuplicateRegions(twoD):
    res = []
    seen = set()
    for x in twoD:
        x_set = frozenset(x)
        if x_set not in seen:
            res.append(x)
            seen.add(x_set)

    return res

def recursiveRegions(x, y, baseSeq, seqLimit, gridSize, clusterSeq):
    if seqLimit == 1:  # We don't want to find neighbouring squares
        clusterSeq.append(baseSeq)
        return

    for x2 in range(x - 1, x + 2):
        if -1 < x2 < gridSize:
            for y2 in range(y - 1, y + 2):
                if -1 < y2 < gridSize:
                    if not (x == x2 and y == y2):
                        if [x2, y2] not in baseSeq:  # Check if coord already in region sequence
                            newBase = baseSeq + [[x2, y2]]
                            if len(newBase) < seqLimit:
                                recursiveRegions(x2, y2, newBase, seqLimit, gridSize, clusterSeq)
                            else:
                                clusterSeq.append(newBase)

def recursiveRegions2(x, y, baseSeq, seqLimit, gridSize, clusterSeq):
    if seqLimit == 1:  # We don't want to find neighbouring squares
        clusterSeq.append(baseSeq)
        clusterSeq.append(getLeftOverRegion(baseSeq, gridSize))
        return

    for x2 in range(x - 1, x + 2):
        if -1 < x2 < gridSize:
            for y2 in range(y - 1, y + 2):
                if -1 < y2 < gridSize:
                    if not (x == x2 and y == y2):
                        if [x2, y2] not in baseSeq:  # Check if coord already in region sequence
                            newBase = baseSeq + [[x2, y2]]
                            if len(newBase) < seqLimit:
                                recursiveRegions(x2, y2, newBase, seqLimit, gridSize, clusterSeq)
                            else:
                                clusterSeq.append(newBase)
                                # if limit is half of the grid size, there's no point of getting the left over region
                                if seqLimit < gridSize*gridSize/2:
                                    clusterSeq.append(getLeftOverRegion(newBase, gridSize))

class ClusterRegions:

    # Given a grid size, return back all possible combinations of regions,
    # given that the region cannot be bigger than 10% of the total grid size.
    # This region will represent the normal heartbeat cluster, and the
    # grid region left over will represent the abnormal heartbeat cluster
    @staticmethod
    def GetClusterRegions(gridSize):
        # maxRegionSize = int(0.1 * gridSize * gridSize)
        # if maxRegionSize > gridSize:
        #     print("maxRegionSize cannot be bigger than gridSize")
        #     return
        clusterRegions = []

        for i in range(1, int(gridSize*gridSize/2+1)):
            iSizedClusterRegions = []
            for x in range(gridSize):
                for y in range(gridSize):
                    regionSequence = []
                    recursiveRegions(x, y, [[x, y]], i, gridSize, regionSequence)
                    if regionSequence is not None:
                        iSizedClusterRegions = iSizedClusterRegions + regionSequence  # merge the lists

            iSizedClusterRegions = convert3dTo2d(iSizedClusterRegions)
            iSizedClusterRegions = removeDuplicateRegions(iSizedClusterRegions)
            clusterRegions += convert2dTo3d(iSizedClusterRegions)

        return clusterRegions

    @staticmethod
    def SaveClusterRegionsToFile(clusterRegions, directory, fileName):
        with open(directory + '/' + fileName, "w") as writer:
            for region in clusterRegions:
                regionStr = str(region)
                writer.write(regionStr[1:len(regionStr)-1] + "\r")

    @staticmethod
    def LoadClusterRegions(filePath):
        clusterRegions = []
        regionsFile = open(filePath, 'r')
        regionLines = regionsFile.readlines()

        for regionStr in regionLines:
            coords = regionStr.split("], ")
            region = []
            for coord in coords:
                region.append([int(coord[1]), int(coord[4])])

            clusterRegions.append(region)

        return clusterRegions

    @staticmethod
    def IsCoordInsideRegion(coord, region):
        seen = set()
        for x in region:
            seen.add(frozenset(x))

        if frozenset(coord) in seen:
            return True
        else:
            return False

#a = convert3dTo2d([[[1, 2], [2, 1]], [[4, 5], [6, 7]]])
#allRegions = ClusterRegions.GetClusterRegions(gridSize=5)
#ClusterRegions.SaveClusterRegionsToFile(allRegions, "C:/Dev/DataSets/ClusterRegions", "GridSize5.txt")
#clusterRegions = ClusterRegions.LoadClusterRegions("C:/Dev/DataSets/ClusterRegions/GridSize6-MaxRegionSize4.txt")