from image_scraper import ImageScraper
from img_preparer import ImageConverter
from convnet import ConvNet, ConvLayer, PoolLayer, FCLayer
import os


def getTrainData(words, numImages):
    # words - list of search terms to use
    # numImages - number of images to get for each category

    imS = ImageScraper("AIzaSyBjRRMtqV4VdybDPjr-tNObKI6qbAukdYE")

    for word in words:
        imS.downloadImages(word, numImages)


#getTrainData(["cat", "giraffe", "house", "tree"], 100)

def importTrainData():
    imC = ImageConverter((150,100))
    
    allImgs = []
    allLabels = []
    #labels and images will be aligned
    
    for folder in os.listdir("trainingData/"):
        for image in os.listdir("trainingData/" + folder):
            allImgs.append(imC.imageToNPArray("trainingData/" +
                                              folder + "/" + image))
            allLabels.append(folder)

    return allImgs, allLabels

#trainData, trainLabels = importTrainData()

def createLayers():
    conv1 = ConvLayer((5,5), 1, 4, (150, 100, 3))
    pool1 = PoolLayer((2,2), 2, conv1.outSize)
    conv2 = ConvLayer((3,3), 1, 4, pool1.outSize)
    pool2 = PoolLayer((3,3), 2, conv2.outSize)
    fcl1 = FCLayer(15, pool2.outSize, prevIsPool=True)
    fcl2 = FCLayer(1, fcl1.outSize)

    layers = [conv1, pool1, conv2, pool2, fcl1, fcl2]

    cNet = ConvNet(layers)

    return cNet
createLayers()
