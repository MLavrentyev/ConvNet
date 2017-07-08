from image_scraper import ImageScraper
from img_preparer import ImageConverter
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


def getTrainData(words, numImages):
    # words - list of search terms to use
    # numImages - number of images to get for each category

    imS = ImageScraper("5847180-524e0715a560748fb4446c545")

    for word in words:
        if not os.path.exists("trainingData/" + word):
            os.makedirs("trainingData/" + word)

        existing_files = os.listdir("trainingData/" + word)
        max_num = -1
        if existing_files:
            max_num = max([int(file.strip("img").strip(".png")) for file in existing_files])

        imS.downloadImages(word, numImages, start = max_num+1)


def importTrainData(categories):
    imC = ImageConverter((150,100))

    allImgs = []
    allLabels = []
    #labels and images will be aligned

    for folder in os.listdir("trainingData/"):
        for image in os.listdir("trainingData/" + folder):
            toAdd = imC.prepImage("trainingData/" + folder + "/" + image)

            for imgArr in toAdd:
                newLabel = np.zeros(len(categories))
                newLabel[categories.index(folder)] = 1

                allLabels.append(newLabel)
                allImgs.append(imgArr)

    print("Images Loaded")
    return allImgs, allLabels


def cnn_function(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 100, 150, 3])

    cnn_layer1 = tf.layers.conv2d(input_layer,
                                  20,
                                  (10, 10),
                                  padding="same",
                                  name="Conv1",
                                  activation=tf.nn.relu)
    pool_layer1 = tf.layers.max_pooling2d(cnn_layer1,
                                          (4, 4), 4,
                                          name="Pool1",)

    cnn_layer2 = tf.layers.conv2d(pool_layer1,
                                  40,
                                  (5, 5),
                                  padding="same",
                                  name="Conv2",
                                  activation=tf.nn.relu)
    pool_layer2 = tf.layers.max_pooling2d(cnn_layer2,
                                          (2, 2), 2,
                                          name="Pool2",)

    cnn_layer3 = tf.layers.conv2d(pool_layer2,
                                  40,
                                  (4, 4),
                                  padding="same",
                                  name="Conv3",)
    pool_layer3 = tf.layers.max_pooling2d(cnn_layer3,
                                          (2, 2), 2,
                                          name="Pool3",)
    flat_pool3 = tf.reshape(pool_layer3, [-1, 7 * 10 * 40])
    dense_layer = tf.layers.dense(flat_pool3,
                                  units=1024,
                                  activation=tf.nn.relu)
    logits = tf.layers.dense(dense_layer,
                             units=2)

    # Finish network architechture

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                               logits=logits)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                   global_step=tf.contrib.framework.get_global_step(),
                                                   learning_rate=0.01,
                                                   optimizer="SGD")
    predictions = {
        "classes": tf.argmax(input=logits, axis=1)
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main():
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = learn.Estimator(model_fn=cnn_function, model_dir="models/mnist_cnn")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    mnist_classifier.fit(x=train_data,
                         y=train_labels,
                         batch_size=50,
                         steps=40000,
                         monitors=[logging_hook])


if __name__ == "__main__":
    #Get training data from Google
    #getTrainData(["school+bus"], 600)

    # Import trianing data into programa
    #trainImgs, trainLabels = importTrainData(["cat" "tree", "lolipop", "house", "shoes", "airplane"])
    #print trainLabels