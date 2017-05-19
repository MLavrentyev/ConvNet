import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce
from functools import reduce

class ConvNet(object):

    def __init__(self, imgShape, c1FiltShape, c1Step, c1NumFilts,
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

        size1 = (np.array(imgShape) - np.array(p1Shape))//p1Step+1
        size2 = (size1-np.array(p2Shape))//p2Step+1
        self.c_h_weights = np.random.random((reduce(lambda x, y: x*y, size2), fcl1Size))*2-1
        self.h_o_weights = np.random.random((fcl1Size, outSize))*2-1

        

    def forwardProp(self, imgArr):
        # imgArr - np array of size (d, w, h)

        self.c1Out = self.forwardConvolution(imgArr, self.c1Filters)
        self.p1Out = self.forwardPooling(self.c1Out, self.p1Shape, self.p1Step)
        self.r1Out = self.applyReLU(self.p1Out)

        self.c2Out = self.forwardConvolution(self.p1Out, self.c2Filters)
        self.p2Out = self.forwardPooling(self.c2Out, self.p2Shape, self.p2Step)
        self.r2Out = self.applyReLU(self.p2Out)

        self.hOut = self.forwardFCL(self.r2Out, self.c_h_weights)
        self.hActOut = self.sigmoidActiv(self.hOut)
        self.out = self.forwardFCL(self.hActOut, self.h_o_weights)
        self.actOut = self.sigmoidActiv(self.out)

        return(self.actOut)

    def forwardConvolution(self, imgArr, filters):
        outArr = np.zeros((len(filters),
                               imgArr.shape[1], imgArr.shape[2]))
        for f in range(len(filters)):
            for channel in imgArr:
                outArr[f] += ndimage.filters.convolve(channel, filters[f], mode='constant', cval=0.0)

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

    def forwardFCL(self, imgArrs, weightsIn):
        flattened = imgArrs.flatten()
        outVals = np.dot(flattened, weightsIn)
        return outVals

    def sigmoidActiv(self, x):
          return 1/(1+np.exp(-x))                         
                                   

cNet = ConvNet((2,3),(2,2), 1, 2,
               (2,2), 1,
               (2,2), 1, 1,
               (1,1), 1,
               15, 1)
imgArr = np.array([[[128., 140, 200],
                    [58, 78, 225],],
                   [[0, 0, 0],
                    [0, 0, 0],],
                   [[110, 58, 170],
                    [200, 225, 255]]])/255
cNet.forwardProp(imgArr)
