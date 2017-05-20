import os

import data_formatter as df
import neural_network as nn
import graphics as graph


DATA_PATH = 'data/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

SAVED_TRAIN = 'train.data'
SAVED_VALIDATION = 'validation.data'
SAVED_TEST = 'test.data'


def create_data():
    train_path = DATA_PATH + TRAIN_FILE
    test_path = DATA_PATH + TEST_FILE

    exclude_columns = ['Ticket', 'Cabin', 'Name']

    train_data, test_data = df.format_data(
        train_path, test_path, exclude_columns, verbose=True)

    train_data = df.format_training_data(train_data)
    print('Creating validation data...')
    train_data, validation_data = df.create_validation_data(
        train_data, verbose=False)

    return train_data, validation_data, test_data


def check_paths():
    train_saved = os.path.exists(SAVED_TRAIN)
    test_saved = os.path.exists(SAVED_TEST)
    validation_saved = os.path.exists(SAVED_VALIDATION)

    return train_saved and test_saved and validation_saved


def main():

    if not check_paths():
        print('Creating data...')
        train_data, validation_data, test_data = create_data()
        df.save_data(train_data, SAVED_TRAIN)
        df.save_data(test_data, SAVED_TEST)
        df.save_data(validation_data, SAVED_VALIDATION)
    else:
        print('Loading data...')
        train_data = df.load_data(SAVED_TRAIN)
        test_data = df.load_data(SAVED_TEST)
        validation_data = df.load_data(SAVED_VALIDATION)

    layers = [7, 6, 1]
    epochs = 50
    batch_size = 200
    legends = []
    costs = []

    network = nn.NeuralNetwork(layers, verbose=True)
    learning_rate = 8
    lambda_value = 0.001
    legends.append(learning_rate)

    train_acc, val_acc, cost_values = network.sgd(
        train_data=train_data,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        lambda_value=lambda_value,
        validation_data=validation_data)
    costs.append(cost_values)

    graph.display_cost_graph(costs, legends)


if __name__ == '__main__':
    main()
