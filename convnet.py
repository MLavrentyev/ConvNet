import numpy as np


class ConvNet(object):

    def __init__(self):
        pass
        
        
class ConvLayer(object):

    def __init__(self, filtShape, filtStep, numFilts):
        # filtShape - tuple (w, h) filter shape
        # filtStep - int step of the filter
        # numFilts - int number of filters

        self.filtShape = filtShape
        self.filtStep = filtStep
        self.numFilts = numFilts

        # Create the filters
        self.filters = [np.random.random(filtShape) for x in range(numFilts)]
