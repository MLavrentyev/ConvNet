import requests
import json
import urllib
import os
import math

class ImageScraper(object):

    def __init__(self, apiKey):
        
        self.baseUrl = "https://pixabay.com/api/"
        self.apiKey = apiKey
        self.per_page = 20
        self.params = {"lang": "en",
                       "image_type": "photo",
                       "orientation": "horizontal",
                       "per_page": self.per_page,
                       "safesearch": "true",
                       }

    def sendRequest(self, query, numImages, start=0):
        # query - string the query to be sent in the Custom Search
        # numImages - int number of images to return, rounded up to nearest <self.per_page>

        start = math.ceil(start/self.per_page)

        self.params["key"] = self.apiKey
        self.params["q"] = urllib.parse.quote_plus(query, safe="")

        imageLinks = []
        for i in range(start, start + math.ceil(numImages/self.per_page)):
            self.params["page"] = i + 1
            print(i+1)

            response = requests.get(self.baseUrl, params=self.params)
            print(response.headers)
            if response.status_code != requests.codes.ok:
                print("The API request has encountered an error: " + str(response.status_code))
                return imageLinks

            response = json.loads(response.content)
            print(response)
            imageLinks.extend([item["previewURL"] for item in response["hits"]])
        # end loop
        
        return imageLinks

    def downloadImages(self, word, numImages, start=0):
        # params - see sendRequest()

        links = self.sendRequest(word, numImages, start=start)

        if links is None:
            return

        if not os.path.exists("trainingData/" + word):
            os.makedirs("trainingData/" + word)
        
        for i in range(len(links)):
            name = "img"
            n = 0
            new_name = name + str(n)
            while os.path.exists("trainingData/" + word + "/" + new_name + ".png"):
                new_name = name + str(n)
                n += 1
            urllib.request.urlretrieve(links[i], "trainingData/" + word + "/" +  new_name + ".png")