import data_formatter as df
import neural_network as nn


DATA_PATH = 'data/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'


def main():
    train_path = DATA_PATH + TRAIN_FILE
    test_path = DATA_PATH + TEST_FILE

    exclude_columns = ['Ticket', 'Cabin', 'Name']

    train_data, test_data = df.format_data(
        train_path, test_path, exclude_columns, verbose=True)

    train_data = df.format_training_data(train_data)
    print('Creating validation data...')
    train_data, validation_data = df.create_validation_data(
        train_data, verbose=True)

    layers = [7, 5, 1]

    network = nn.NeuralNetwork(layers, verbose=True)
    batch_size = 50
    epochs = 20
    learning_rate = 0.3

    network.sgd(
        train_data=train_data,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        validation_data=validation_data)


if __name__ == '__main__':
    main()
