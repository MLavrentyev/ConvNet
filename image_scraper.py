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

    def sendRequest(self, query):
        # query - string the query to be sent in the Custom Search
        self.params["q"] = query
        self.params["cx"] = "002614461317739606024:pdrv0eu2ihq"
        
        response = requests.get(self.baseUrl, params=self.params).content       
        response = json.loads(response)

        imageLinks = [val["image"]["thumbnailLink"]
                      for val in response["items"]]
            
        return imageLinks

    def downloadImages(self, word):
        links = self.sendRequest(word)

        if not os.path.exists(word):
            os.makedirs(word)
        
        for i in range(len(links)):
            urllib.request.urlretrieve(links[i],
                                       word + "/" + "img" + str(i) + ".png")
        

imS = ImageScraper("AIzaSyBjRRMtqV4VdybDPjr-tNObKI6qbAukdYE")
imS.downloadImages("orchid")
