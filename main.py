from image_scraper import ImageScraper
from img_preparer import ImageConverter
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


tf.logging.set_verbosity(tf.logging.INFO)


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


# ------------Neural Net Stuff-----------------------

def conv_layer(input, channels_in, channels_out):
    W = tf.Variable(tf.random_uniform([5, 5, channels_in, channels_out]))
    b = tf.Variable(tf.constant(0.1, shape=[channels_out]))
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
    activation = tf.nn.relu(conv + b)

    return activation


def fc_layer(input, channels_in, channels_out):
    W = tf.Variable(tf.random_uniform([channels_in, channels_out]))
    b = tf.Variable(tf.constant(0.1, shape=[channels_out]))
    ff = tf.matmul(input, W) + b
    activation = tf.nn.relu(ff)

    return activation


def cnn_function(x, y):
    x_2d_img = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("conv1"):
        conv1 = conv_layer(x_2d_img, 1, 20)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("conv2"):
        conv2 = conv_layer(pool1, 20, 40)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        flattened = tf.reshape(pool2, [-1, 7*7*40])

    fcl1 = fc_layer(flattened, 7*7*40, 1024)
    fcl2 = fc_layer(fcl1, 1024, 512)
    logits = fc_layer(fcl2, 512, 10)

    return logits


def main(unused_argv):
    sess = tf.Session()

    # Load dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # Create network
    logits = cnn_function(x, y)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("tmp/mnist_demo/1")
    writer.add_graph(sess.graph)

    # Train
    for i in range(2000):
        batch = mnist.train.next_batch(100)

        # Report accuracy every 100 steps
        if i%100 == 0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y:batch[1]})
            print("Step: %d, Training accuracy: %d" % (i, train_accuracy))

        # Run training step
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


if __name__ == "__main__":
    tf.app.run()