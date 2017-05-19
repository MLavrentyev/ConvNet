from PIL import Image
import numpy as np

class ImageConverter(object):

    def __init__(self, imageSize):
        # imageSize - tuple (w,h) to normalize images to

        self.imgWidth = imageSize[0]
        self.imgHeight = imageSize[1]
        

    def imageToNPArray(self, imgPath):
        # imgPath - string path to the rgb image
        img = Image.open(imgPath)

        if img.size != (self.imgWidth, self.imgHeight):
            img = img.resize((self.imgWidth, self.imgHeight), Image.LANCZOS)

        img = img.convert(mode='RGB')

        rPix = np.array(list(img.getdata(band=0))).reshape(self.imgWidth,
                                                           self.imgHeight)
        gPix = np.array(list(img.getdata(band=1))).reshape(self.imgWidth,
                                                           self.imgHeight)
        bPix = np.array(list(img.getdata(band=2))).reshape(self.imgWidth,
                                                           self.imgHeight)

        return np.array([rPix, gPix, bPix])

    def scaleImage(self, imgArr):
        return imgArr/255*2 - 1

    def prepImage(self, imgPath):
        arr = self.imageToNPArray(imgPath)
        scaledArr = self.scaleImage(arr)

        return scaledArr
