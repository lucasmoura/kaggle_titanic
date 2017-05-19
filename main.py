import data_formatter as df
import neural_network as nn
import graphics as graph


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
    epochs = 30
    batch_size = 200
    legends = []
    costs = []

    network = nn.NeuralNetwork(layers, verbose=True)
    learning_rate = 6
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
