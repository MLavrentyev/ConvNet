import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce
from functools import reduce

class ConvNet(object):

    def __init__(self, imgShape, numChannels,
                 c1FiltShape, c1NumFilts,
                 p1Shape, p1Step,
                 c2FiltShape, c2NumFilts,
                 p2Shape, p2Step,
                 fcl1Size, outSize):

        self.inImgShape = imgShape

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

        self.c_h_weights = np.random.random((np.prod(size2)*numChannels, fcl1Size))*2-1
        self.h_o_weights = np.random.random((fcl1Size, outSize))*2-1

        self.preFclSize = size2*numChannels
        self.fcl1Size = fcl1Size
        self.outSize = outSize
        

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
        outArr = np.zeros((len(filters), imgArr.shape[1], imgArr.shape[2]))
        
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

    def calcSquaredError(self, output, rOutput):
        return np.sum(0.5 * np.power(rOutput - output, 2))

    def backProp(self, inDatum, rOut):
        # inDatum is the in data for the network (1 p.)
        # rOut is the out data for the netowrk (1 p.)

        nOut = self.forwardProp(inDatum)

        
        # Calculate gradient with weights of 1st FCL --> out
        dC_dfcl_w2 = np.zeros(self.h_o_weights.shape)
        for o in range(self.outSize):
            dC_dfcl_w2[:,o] = np.multiply((nOut[o]-rOut[o])*nOut[o]*(1-nOut[o]), self.hActOut)
            #TODO: calculate deltas


        # Calculate gradient with weights of finLayer --> 1st FCL
        dC_dfcl_w1 = np.zeros(self.c_h_weights.shape)
        for h in range(self.fcl1Size):
            dC_dfcl_w1[:,h] = np.multiply()#TODO: Finish this
    
                                   

