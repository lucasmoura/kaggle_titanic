import data_formatter as df


DATA_PATH = 'data/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'


def main():
    train_path = DATA_PATH + TRAIN_FILE
    test_path = DATA_PATH + TEST_FILE

    df.format_data(train_path, test_path)


if __name__ == '__main__':
    main()
