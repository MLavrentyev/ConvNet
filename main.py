from image_scraper import ImageScraper


def getTrainData(words, numImages):
    # words - list of search terms to use
    # numImages - number of images to get for each category

    imS = ImageScraper("AIzaSyBjRRMtqV4VdybDPjr-tNObKI6qbAukdYE")

    for word in words:
        imS.downloadImages(word, numImages)

