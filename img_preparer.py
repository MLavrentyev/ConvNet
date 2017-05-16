import PIL
import numpy as np

class ImageConverter(object):

    def __init__(self, imageSize):
        # imageSize - tuple (w,h) to normalize images to

        self.imgWidth = imageSize[0]
        self.imgHeight = imageSize[1]
        

    def imageToNP(self, image):
        pass

    def resizeImage(self, imgArr):
        pass
