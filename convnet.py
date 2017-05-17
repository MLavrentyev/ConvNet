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

    def forwardProp(self, imgArr):
        # imgArr - np array of size (d, w, h)

        # Apply 1st convolution
        self.c1Out = np.zeros((len(self.c1Filters),
                               imgArr.shape[1], imgArr.shape[2]))
        for f in range(len(self.c1Filters)):
            for channel in imgArr:
                self.c1Out[f] = ndimage.filters.convolve(channel,
                                                         self.c1Filters[f],
                                                         mode='constant',
                                                         cval=0.0)
        #End 1st Convolution
        newSizes = (self.c1Out.shape[0],
                    int((self.c1Out.shape[1]-self.p1Shape[0])/self.p1Step)+1,
                    int((self.c1Out.shape[2]-self.p1Shape[1])/self.p1Step)+1)

        #Apply 1st pooling
        self.p1Out = np.zeros(newSizes)
        print(self.c1Out)
                               
        ySpots = np.array([y for y in range(0, self.c1Out.shape[1]+1-self.p1Shape[0], self.p1Step)])
        xSpots = np.array([x for x in range(0, self.c1Out.shape[2]+1-self.p1Shape[1], self.p1Step)])
        print(xSpots, ySpots)

        for chan in range(newSizes[0]):
            ixmesh = np.ix_([chan], np.arange(len(ySpots)), np.arange(len(xSpots)))
            
            self.p1Out[ixmesh] = [[np.amax(self.c1Out[chan, r:r+self.p1Shape[1], c:c+self.p1Shape[0]]) for c in xSpots] for r in ySpots]

        print(self.p1Out)

            


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
