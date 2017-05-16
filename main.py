from image_scraper import ImageScraper
from img_preparer import ImageConverter
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

trainData, trainLabels = importTrainData()

