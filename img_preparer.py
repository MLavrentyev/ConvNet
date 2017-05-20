from PIL import Image
import numpy as np

class ImageConverter(object):

    def __init__(self, imageSize):
        # imageSize - tuple (w,h) to normalize images to

        self.imgWidth = imageSize[0]
        self.imgHeight = imageSize[1]
        

    def imageToNPArray(self, img):
        # img - PIL Image object

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
        img = Image.open(imgPath)

        augmentedIms = self.createAdjustedImages(img)
        
        allIms = []
        for im in augmentedIms:
            arr = self.imageToNPArray(im)
            allIms.append(self.scaleImage(arr))

        return allIms

    def createAdjustedImages(self, img):
        # Returns the original and various rotated and shifted copies.
        newData = [img]

        newData.append(img.rotate(90))
        newData.append(img.rotate(180))
        newData.append(img.rotate(270))

        #for i in newData:
            #newData.append(i.crop((37, 25, 113, 75)).resize((150,100)))

        return newData
                           
