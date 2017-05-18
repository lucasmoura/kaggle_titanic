import tensorflow as tf
from math import sqrt


class NeuralNetwork:

    def __init__(self, layers, verbose=False):
        """
        This method will be used to initialize the Neural Network.
        The first parameter, layers will state how much neurons each
        layer in the network will have.

        For example, if layers is [9, 5, 1], it means that the input
        layer will 9 neuros, the hidden layer 5 and the outpur layer 1.
        """

        self.sess = tf.Session()
        self.verbose = verbose

        self.init_biases(layers)
        self.init_weights(layers)

    def init_biases(self, layers):
        self.biases = []
        input_layer = layers[0]

        for index, layer in enumerate(layers[1:]):
            bias = tf.Variable(
                tf.random_normal(
                    shape=[layer, 1],
                    mean=0,
                    stddev=(1 / sqrt(input_layer))),
                name='b{}'.format(index))
            self.biases.append(bias)
            input_layer = layer

    def init_weights(self, layers):
        self.weights = []
        index = 1

        for s1, s2 in zip(layers[:-1], layers[1:]):
            weight = tf.Variable(
                tf.random_normal(
                    shape=[s1, s2],
                    mean=0,
                    stddev=(1 / sqrt(s1))),
                name='w{}'.format(index))
            self.weights.append(weight)
            index += 1

    def print_biases(self):
        for bias in self.biases:
            print('Bias with shape: {} and values:\n {}'.format(
                self.sess.run(tf.shape(bias)),
                self.sess.run(bias)))
        print()

    def print_weights(self):
        for weight in self.weights:
            print('Weight with shape: {} and values:\n {}'.format(
                self.sess.run(tf.shape(weight)),
                self.sess.run(weight)))
        print()

    def sgd(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.verbose:
            print('\nDisplaying biases and weights before sgd...')
            self.print_biases()
            self.print_weights()
