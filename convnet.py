import numpy as np


class ConvNet(object):
    
    LEARNING_RATE = 0.5
    
    def __init__(self, layers):

        self.layers = layers
        
        
        
class ConvLayer(object):

    def __init__(self, filtShape, filtStep, numFilts, prevOutSize):
        # filtShape - tuple (w, h) filter shape
        # filtStep - int step of the filter
        # numFilts - int number of filters
        # prevOutSize - tuple (w, h, d) of the output of the previous layer
        #               if first, it's the input size

        self.filtShape = filtShape[0], filtShape[1], prevOutSize[2]
        self.filtStep = filtStep
        self.numFilts = numFilts

        self.outSize = ((prevOutSize[0] - filtShape[0])//filtStep + 1,
                        (prevOutSize[1] - filtShape[1])//filtStep + 1, 1)

        # Create the filters
        self.filters = [np.random.random(filtShape)*2 - 1
                        for x in range(numFilts)]

class PoolLayer(object):

    def __init__(self, poolShape, poolStep, prevOutSize):
        # poolShape - tuple (w, h) pooling shape
        # poolStep - int step of the pooling layer
        # prevOutSize - tuple (w, h, d) of the output of the previous layer

        self.poolShape = poolShape
        self.poolStep = poolStep

        self.outSize = ((prevOutSize[0] - poolShape[0])//poolStep + 1,
                        (prevOutSize[1] - poolShape[1])//poolStep + 1, 1)
        

class FCLayer(object):

    def __init__(self, numNeurons, prevOutSize, prevIsPool=False):
        # numNeurons - int number of neurons
        # prevIsPool - specifies if the previous layer is a pool layer
        # prevOutSize - [tuple (w,h) True | int False ]

        pass

        self.numNeurons = numNeurons
        self.outSize = numNeurons

        if prevIsPool:
            self.inWeights = np.random.random((prevOutSize[0]*prevOutSize[1]*
                                               prevOutSize[2], numNeurons))*2-1
        else:
            self.inWeights = np.random.random((prevOutSize, numNeurons))*2-1
        
