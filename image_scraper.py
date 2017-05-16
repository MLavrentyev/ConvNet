import requests
import json
import urllib.request
import os

class ImageScraper(object):

    def __init__(self, apiKey):
        
        self.baseUrl = "https://www.googleapis.com/customsearch/v1"
        self.apiKey = apiKey

        self.params = {"searchType": "image",
                       "imgColorType": "color",
                       "imgType": "photo",
                       "filter": "0",
                       "safe": "medium",
                       "key": apiKey}

    def sendRequest(self, query, numImages):
        # query - string the query to be sent in the Custom Search
        # numImages - int number of images to return, rounded down to nearest 10

        self.params["q"] = query
        self.params["cx"] = "002614461317739606024:pdrv0eu2ihq"

        imageLinks = []
        for i in range(int(numImages/10)):
            print(i)
            self.params["start"] = i+1

            response = requests.get(self.baseUrl, params=self.params).content       
            response = json.loads(response)
            print(response)
            imageLinks.extend([val["image"]["thumbnailLink"]
                          for val in response["items"]])
        # end loop
        
        return imageLinks

    def downloadImages(self, word, numImages):
        # params - see sendRequest()

        links = self.sendRequest(word, numImages)

        if not os.path.exists(word):
            os.makedirs(word)
        
        for i in range(len(links)):
            urllib.request.urlretrieve(links[i],
                                       word + "/" + "img" + str(i) + ".png")
        

imS = ImageScraper("AIzaSyBjRRMtqV4VdybDPjr-tNObKI6qbAukdYE")
imS.downloadImages("orchid", 24)
