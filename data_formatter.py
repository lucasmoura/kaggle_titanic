import pandas


def create_dataframe(data_path):
    return pandas.read_csv(data_path)


def combine_data(data1, data2):
    return [data1, data2]


def drop_columns(dataframe, exclude_columns):
    return dataframe.drop(exclude_columns, axis=1)
    

def format_data(train_path, test_path, exclude_columns=None):
    train_data = create_dataframe(train_path)
    test_data = create_dataframe(test_path)

    if exclude_columns:
        train_data = drop_columns(train_data, exclude_columns)
        test_data = drop_columns(test_data, exclude_columns)

    return train_data, test_data
