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

        #Apply 1st pooling
        self.p1Out = np.zeros((self.c1Out.shape[0],
                               
        for channel in self.c1Out:
            self.p1Out[] = None                          

            


cNet = ConvNet((2,2), 1, 2,
               (1,1), 1,
               (2,2), 1, 1,
               (1,1), 1,
               15, 4)
imgArr = np.array([[[1., 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[2, 4, 6],
                    [8, 10, 12],
                    [14, 16, 18]]])/20
cNet.forwardProp(imgArr)
