from image_scraper import ImageScraper
from img_preparer import ImageConverter
from convnet import ConvNet
import os
import numpy as np


def getTrainData(words, numImages):
    # words - list of search terms to use
    # numImages - number of images to get for each category

    imS = ImageScraper("AIzaSyBjRRMtqV4VdybDPjr-tNObKI6qbAukdYE")

    for word in words:
        imS.downloadImages(word, numImages)


#getTrainData(["lollipop", "stop sign", "shoes"], 100)

def importTrainData():
    imC = ImageConverter((150,100))

    labelToNum = ["cat", "giraffe", "house", "lollipop", "shoes",
                  "stop sign", "tree"]
    
    allImgs = []
    allLabels = []
    #labels and images will be aligned
    
    for folder in os.listdir("trainingData/"):
        for image in os.listdir("trainingData/" + folder):
            toAdd = imC.prepImage("trainingData/" + folder + "/" + image)

            for imgArr in toAdd:
                newLabel = np.zeros(7)
                newLabel[labelToNum.index(folder)] = 1
                
                allLabels.append(newLabel)
                allImgs.append(imgArr)

    print("Images Loaded")
    return allImgs, allLabels

trainData, trainLabels = importTrainData()
trainData = np.array(trainData)


cNet = ConvNet((100,150), 3,
               (4,4), 3,
               (3,3), 3,
               (4,4), 3,
               (4,4), 4,
               20, 7)
#print(cNet.forwardProp(trainData[0]))

#print(trainLabels[0])
