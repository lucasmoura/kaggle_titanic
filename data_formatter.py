import pandas


def create_dataframe(data_path):
    return pandas.read_csv(data_path)


def combine_data(data1, data2):
    return data1.append(data2)


def drop_columns(dataframe, exclude_columns):
    return dataframe.drop(exclude_columns, axis=1)


def add_title_column(dataframe):
    dataframe['Title'] = dataframe.Name.str.extract(' (\w+)\.', expand=False)
    return dataframe


def split_dataframe(dataframe, split_dataframe_len):
    return dataframe[:split_dataframe_len], dataframe[split_dataframe_len:]


def format_data(train_path, test_path, exclude_columns=None, verbose=False):
    train_data = create_dataframe(train_path)
    test_data = create_dataframe(test_path)

    if verbose:
        print('Train data shape[0]: {}'.format(train_data.shape[0]))
        print('Test data shape[0]: {}'.format(test_data.shape[0]))

    train_data_len = train_data.shape[0]
    combined_data = combine_data(train_data, test_data)

    if exclude_columns:
        combined_data = drop_columns(combined_data, exclude_columns)

    combined_data = add_title_column(combined_data)

    train_data, test_data = split_dataframe(combined_data, train_data_len)

    if verbose:
        print('\nTrain data shape[0]: {}'.format(train_data.shape[0]))
        print('Test data shape[0]: {}'.format(test_data.shape[0]))

    return train_data, test_data
