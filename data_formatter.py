import pandas


def create_dataframe(data_path):
    return pandas.read_csv(data_path)


def combine_data(data1, data2):
    return [data1, data2]
    

def format_data(train_path, test_path):
    training_data = create_dataframe(train_path)
    test_data = create_dataframe(test_path)

    combined_data = combine_data(training_data, test_data)


