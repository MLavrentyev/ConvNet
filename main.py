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

def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[channels_out]))
        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
        activation = tf.nn.relu(conv + b)

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("act", activation)
        return activation


def fc_layer(input, channels_in, channels_out, name="fcl"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        ff = tf.matmul(input, W) + b
        activation = tf.nn.relu(ff)

        return activation


def cnn_function(x, y):
    x_2d_img = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = conv_layer(x_2d_img, 1, 32, name="conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = conv_layer(pool1, 32, 64, name="conv2")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    flattened = tf.reshape(pool2, [-1, 7*7*64])

    fcl1 = fc_layer(flattened, 7*7*64, 1024, name="fcl1")
    logits = fc_layer(fcl1, 1024, 10, name="logits")

    return logits


def main(unused_argv):
    sess = tf.Session()

    # Load dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    # Create network
    logits = cnn_function(x, y)
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("xent", cross_entropy)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # Initialize variables and tensorboard summaries
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("tmp/mnist_demo/6")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    # Create a saver to save the trained model
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25,
                           name="MNIST_set")
    # Train
    for i in range(5000):
        batch = mnist.train.next_batch(100)

        # Report accuracy every 100 steps
        if i%10 == 0:
            s = sess.run(merged_summary, feed_dict={x: batch[0], y:batch[1]})
            writer.add_summary(s, i)
        if i%500 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            print("Step: %d, Training accuracy: %.2f" % (i, train_accuracy))
            saver.save(sess, "trained_models/mnist", global_step=i)

        # Run training step
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

    saver.save(sess, "trained_models/mnist", global_step=i)

if __name__ == "__main__":
    tf.app.run()