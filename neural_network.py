import random

import tensorflow as tf
import numpy as np

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
        self.layers = layers

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

    def initialize_weights_and_biases(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def create_placeholders(self, batch_size, num_features):
        x = tf.placeholder(
            tf.float32, shape=None)
        y = tf.placeholder(
            tf.float32, shape=None)

        return (x, y)

    def create_batches(self, train_data, batch_size):
        random.shuffle(train_data)
        batches = [train_data[offset:offset+batch_size]
                   for offset in range(0, len(train_data), batch_size)]
        return batches

    def unify_batch(self, batch):
        data_batch, prediction_batch = batch[0]

        for data, prediction in batch[1:]:
            data_batch = np.concatenate((data_batch, data), axis=1)
            prediction_batch = np.concatenate(
                (prediction_batch, prediction), axis=1)

        return(data_batch, prediction_batch.T)

    def feedforward(self, input_data):
        a0 = input_data
        z1 = tf.matmul(tf.transpose(self.weights[0]), a0)
        a1 = tf.sigmoid(z1)

        z2 = tf.matmul(tf.transpose(self.weights[1]), a1)
        a2 = tf.sigmoid(z2)

        return tf.transpose(a2)

    def accuracy(self, data, prediction):
        x = tf.placeholder(tf.float32)
        result = self.feedforward(x)

        nn_prediction = self.sess.run(result, {x: data})
        nn_prediction = nn_prediction >= 0.5

        a = tf.placeholder(tf.bool)
        b = tf.placeholder(tf.bool)
        accuracy_metric = tf.contrib.metrics.accuracy(a, b)
        return self.sess.run(accuracy_metric,
                             {a: nn_prediction, b: prediction})

    def sgd(self, *, train_data, batch_size, epochs, learning_rate,
            validation_data):
        self.initialize_weights_and_biases()
        num_features = train_data[0][0].shape[0]
        x, y = self.create_placeholders(batch_size, num_features)

        training_accuracies, validation_accuracies = [], []
        loss_values = []
        tdf, pdf = self.unify_batch(train_data)

        if validation_data:
            data, prediction = self.unify_batch(validation_data)
            prediction = prediction == 1
            pdf_bool = pdf == 1

        if self.verbose:
            print('Stochastic gradient descent will be performed with:')
            print('Num training_data: {}'.format(len(train_data)))
            print('Num epochs: {}'.format(epochs))
            print('Batch size: {}'.format(batch_size))
            print('\nDisplaying biases and weights before sgd...')
            self.print_biases()
            self.print_weights()

        for epoch in range(epochs):
            batches = self.create_batches(train_data, batch_size)

            for batch in batches:
                data_batch, prediction_batch = self.unify_batch(batch)

                output = self.feedforward(x)
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=output,
                        labels=y))
                train_step = tf.train.GradientDescentOptimizer(
                    learning_rate).minimize(loss)

                self.sess.run(
                    train_step,
                    feed_dict={x: data_batch, y: prediction_batch})

            loss_value = self.sess.run(loss, feed_dict={x: tdf, y: pdf})
            loss_values.append(loss_value)

            if validation_data:
                training_accuracy = self.accuracy(tdf, pdf_bool)
                validation_accuracy = self.accuracy(data, prediction)

                print('Epoch {}...'.format(epoch))
                print('Accuracy on validation data: {}'.format(
                    validation_accuracy))
                print('Accuracy on training data: {}'.format(
                    training_accuracy))
                print('Cost value for training data: {}'.format(loss_value))
                print()

                training_accuracies.append(training_accuracy)
                validation_accuracies.append(validation_accuracy)

        return (training_accuracy, validation_accuracy, loss_values)
