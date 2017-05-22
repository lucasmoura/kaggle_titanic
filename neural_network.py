import random
import utils
import numpy as np
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

    def initialize_computing_graph(self, x, y, learning_rate, lambda_value):
        output = self.feedforward(x)
        regularizer = tf.nn.l2_loss(self.weights[0]) + tf.nn.l2_loss(
            self.weights[1])
        regularizer = lambda_value * regularizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=output, labels=y))

        loss_regularized = tf.add(loss, regularizer)

        train_step = tf.train.MomentumOptimizer(
            learning_rate, 0).minimize(loss_regularized)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        return (loss_regularized, train_step)

    def feedforward(self, input_data):
        a0 = input_data
        z1 = tf.add(tf.matmul(tf.transpose(self.weights[0]), a0),
                    self.biases[0])
        a1 = tf.nn.sigmoid(z1)

        z2 = tf.add(tf.matmul(tf.transpose(self.weights[1]), a1),
                    self.biases[1])
        return tf.transpose(z2)

    def predict(self, input_data):
        x = tf.placeholder(tf.float32)
        output = tf.nn.softmax(self.feedforward(x))

        output = self.sess.run(tf.argmax(output, 1),
                               {x: input_data})

        return output.astype(int)

    def accuracy(self, data, prediction):
        x = tf.placeholder(tf.float32)
        result = tf.argmax(tf.nn.softmax(self.feedforward(x)), 1)

        nn_prediction = self.sess.run(result, {x: data})

        a = tf.placeholder(tf.bool)
        b = tf.placeholder(tf.bool)
        accuracy_metric = tf.contrib.metrics.accuracy(a, b)
        return self.sess.run(accuracy_metric,
                             {a: nn_prediction, b: prediction})

    def prepare_batches(self, train_data, batch_size):
        train_batches = utils.create_batches(train_data, batch_size)
        batches = []

        for batch in train_batches:
            data_batch, prediction_batch = utils.unify_batch(batch)
            batches.append((data_batch, prediction_batch))

        return batches

    def sgd(self, *, train_data, batch_size, epochs, learning_rate,
            lambda_value, validation_data):
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)
        training_accuracies, validation_accuracies = [], []
        loss_values = []

        best_validation = -1
        count_validation = 0
        epoch = 1

        if validation_data:
            data, prediction = utils.unify_batch(validation_data)
            prediction = np.argmax(prediction, 1)

            tdf, pdf = utils.unify_batch(train_data)
            pdf_bool = np.argmax(pdf, 1)

        if self.verbose:
            print('Stochastic gradient descent will be performed with:')
            print('Num training_data: {}'.format(len(train_data)))
            print('Num epochs: {}'.format(epochs))
            print('Batch size: {}'.format(batch_size))

        loss, train_step = self.initialize_computing_graph(
            x, y, learning_rate, lambda_value)

        batches = self.prepare_batches(train_data, batch_size)
        saver = tf.train.Saver()

        while(True):
            random.shuffle(batches)
            for batch in batches:
                data_batch, prediction_batch = batch

                self.sess.run(
                    train_step,
                    feed_dict={x: data_batch, y: prediction_batch})

            loss_value = self.sess.run(loss, feed_dict={x: tdf, y: pdf})
            loss_values.append(loss_value)

            if validation_data:
                training_accuracy = self.accuracy(tdf, pdf_bool)
                validation_accuracy = self.accuracy(data, prediction)

                if validation_accuracy > best_validation:
                    best_validation = validation_accuracy
                    count_validation = 0
                    saver_path = saver.save(self.sess, '/tmp/model.ckpt')
                else:
                    count_validation += 1

                print('Epoch {}...'.format(epoch))
                print('Learning rate: {}'.format(learning_rate))
                print('Count for decrese learning rate: {}'.format(
                    count_validation))
                print('Accuracy on validation data: {}'.format(
                    validation_accuracy))
                print('Accuracy on training data: {}'.format(
                    training_accuracy))
                print('Cost value for training data: {}'.format(loss_value))
                print()

                training_accuracies.append(training_accuracy)
                validation_accuracies.append(validation_accuracy)
                epoch += 1

                if count_validation == 20:
                    break

        print('Best achieved accuracy: {}'.format(best_validation))
        saver.restore(self.sess, saver_path)
        return (training_accuracy, validation_accuracy, loss_values)
