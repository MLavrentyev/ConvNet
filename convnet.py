import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce

class ConvNet(object):

    def __init__(self, c1FiltShape, c1Step, c1NumFilts,
                 p1Shape, p1Step,
                 c2FiltShape, c2Step, c2NumFilts,
                 p2Shape, p2Step,
                 fcl1Size, outSize):

        self.c1Filters = [np.random.random(c1FiltShape)*2-1
                          for x in range(c1NumFilts)]

        self.c2Filters = [np.random.random(c2FiltShape)*2-1
                          for x in range(c2NumFilts)]

        self.p1Shape = p1Shape
        self.p1Step = p1Step

        self.p2Shape = p2Shape
        self.p2Step = p2Step

    def forwardProp(self, imgArr):
        # imgArr - np array of size (d, w, h)

        self.c1Out = self.forwardConvolution(imgArr, self.c1Filters)
        self.p1Out = self.forwardPooling(self.c1Out, self.p1Shape, self.p1Step)
        self.r1Out = self.applyReLU(self.p1Out)

        self.c2Out = self.forwardConvolution(self.p1Out, self.c2Filters)
        self.p2Out = self.forwardPooling(self.c2Out, self.p2Shape, self.p2Step)
        self.r2Out = self.applyReLU(self.p2Out)

    def forwardConvolution(self, imgArr, filters):
        outArr = np.zeros((len(filters),
                               imgArr.shape[1], imgArr.shape[2]))
        for f in range(len(filters)):
            for channel in imgArr:
                outArr[f] = ndimage.filters.convolve(channel, filters[f], mode='constant', cval=0.0)

        return outArr

    def forwardPooling(self, imgArrs, pShape, pStep):
        newSizes = (imgArrs.shape[0],
                    int((imgArrs.shape[1]-pShape[0])/pStep)+1,
                    int((imgArrs.shape[2]-pShape[1])/pStep)+1)
        
        outArr = np.zeros(newSizes)
                               
        ySpots = np.array([y for y in range(0, imgArrs.shape[1]+1-pShape[0], pStep)])
        xSpots = np.array([x for x in range(0, imgArrs.shape[2]+1-pShape[1], pStep)])

        for chan in range(newSizes[0]):
            ixmesh = np.ix_([chan], np.arange(len(ySpots)), np.arange(len(xSpots)))
            
            outArr[ixmesh] = [[np.amax(imgArrs[chan, r:r+pShape[1], c:c+pShape[0]]) for c in xSpots] for r in ySpots]

        return outArr

    def applyReLU(self, imgArrs):
        zeroArr = np.zeros(imgArrs.shape)

        return np.maximum(zeroArr, imgArrs)



cNet = ConvNet((2,2), 1, 2,
               (2,2), 1,
               (2,2), 1, 1,
               (1,1), 1,
               15, 4)
imgArr = np.array([[[1., 2, 3],
                    [4, 5, 6],],
                   [[0, 0, 0],
                    [0, 0, 0],],
                   [[2, 4, 6],
                    [8, 10, 12]]])/20
cNet.forwardProp(imgArr)
