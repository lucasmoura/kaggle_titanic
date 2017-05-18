import unittest

import neural_network as nn


class NeuralNetworkTest(unittest.TestCase):

    def setUp(self):
        layers = [3, 2, 1]
        self.neural_net = nn.NeuralNetwork(layers)

    def test_create_batches(self):
        train_data = list(range(25))
        batch_size = 5

        batches = self.neural_net.create_batches(
            train_data, batch_size)

        self.assertEqual(len(batches), 5)

        for batch in batches:
            self.assertEqual(len(batch), batch_size)

        train_data = list(range(891))
        batch_size = 10

        batches = self.neural_net.create_batches(
            train_data, batch_size)

        self.assertEqual(len(batches), 90)

        for batch in batches[:-1]:
            self.assertEqual(len(batch), batch_size)

        self.assertEqual(len(batches[-1]), 1)


if __name__ == '__main__':
    unittest.main()
