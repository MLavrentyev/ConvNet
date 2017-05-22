# ConvNet
This is a convolutional neural network application in Python for image classification. 

## Dependencies
NumPy (latest)<br>
SciPy (latest)<br>
Scikit-Image (latest)<br> 
Pillow (4.1.1 or 4.0.0, **not** 4.1.0)<br>


## Training Data
For training data, the `ImageScraper` class pulls images off of Google Images using the Google Custom Search API. It then saves those images (in the quantity specified rounded down to the nearest 10) in the folder corresponding to the search term, within a folder called `trainingData`. So, for example, if you were to use the term **cat**, and specify 100 images to be pulled, you would have a folder of cat pictures under the directory `trainingData/cat`. To increase how much data there is for training, the `ImageConverter` class, when turning the data into a numpy array, also adds rotated versions (90, 180, and 270 degrees). This quadruples how much training data there is.

To note: Currently my Google API Key is used for the Custom Search requests. Please, if you use this program, *obtain your own API key*. There is a limit to the requests, and it is free to obtain a key from Google. Just switch out the key in `main.py` in the `getTrainingData()` function to use your own.
