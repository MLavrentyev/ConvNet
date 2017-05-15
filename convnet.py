import numpy as np


class ConvNet(object):

    def __init__(self, nChannels, nConvFilt, convFiltSize, convStep,
                 poolSize, poolStep, hidLayerSize, outLayerSize):
        # nChannels - int # of color channels in images
        # nConvFilt - int # of convoultional filters
        # convFiltSize - tuple conv size as (w, h)
        # convStep - int step of convFilter
        #
        # poolSize - tuple pool size as (w, h)
        # poolStep - int step of pooling layer
        #
        # hidLayerSize - int hidden fully-connected layer size
        # outLayerSize - int output layer size

        self.numChan = nChannels
        self.numConvFilt = nConvFilt
        self.convFiltSize = convFiltSize
        self.convStep = convStep
        convFilts = [np.random.normal(size=convFiltSize) \
                     for x in range(nConvFilt)]

        self.poolSize = poolSize
        self.poolStep = poolStep

        self.postConvSize 

        self.hidLayerSize = hidLayerSize
        self.c_h_weights = np.random.normal(size=(self.postConvSize,
                                                  hidLayerSize))
        
        self.outLayerSize = outLayerSize
        self.h_o_weights = np.random.normal(size=(hidLayerSize,
                                                  outLayerSize))

    def applyConvolution(self, imgArr):
        # imgArr - array of 2d np arrays for each color channel

        pass
        
        
