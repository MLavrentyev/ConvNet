import requests

class ImageScraper(object):

    def __init__(self):
        
        self.baseUrl = "https://www.googleapis.com/customsearch/v1"
        self.apiKey = "AIzaSyBjRRMtqV4VdybDPjr-tNObKI6qbAukdYE"

        self.params = {"searchType": "image",
                       "imgColorType": "color",
                       "filter": "0",
                       "safe": "medium"}

    def sendRequest(self, query):
        # query - string the query to be sent in the Custom Search
        self.params["q"]: query
