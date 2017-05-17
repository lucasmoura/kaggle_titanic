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


"""
    The Sex column in the dataframe has the as values the actual gender name of
    a passager. In order to use that information, it will be required to change
    these values to numerical ones:

    0 = female
    1 = male
"""
def format_sex_column(dataframe):
    dataframe.loc[dataframe.Sex == 'female', 'Sex'] = 0
    dataframe.loc[dataframe.Sex == 'male', 'Sex'] = 1


"""
    Every row on training data:

    * PassagerId: The identifier of a given passager;
    * Survived: If the passager survived or not (0 = No, 1 = Yes);
    * Pclass: The ticket class for a passager (1 = 1st, 2 = 2nd, 3 = 3rd);
    * Name: The passage's name;
    * Sex: The passager sex (male or female);
    * Age: The age in years of a passager;
    * SibSp: Number of siblings and spouses aboard;
    * Parch: Number of parents and children aboard;
    * Ticket: The ticker number of the passager;
    * Fare: The amount of money the passager has paid for the ticket;
    * Cabin: The passager's cabin number;
    * Embarked: Where the passager embarked
                (C = Cherbourgh, Q = Queenstown, S = Southampton)

    This function will them be used to format the data for it to be fit to use
    on a neural network.
"""
def format_data(train_path, test_path, exclude_columns=None, verbose=False):
    train_data = create_dataframe(train_path)
    test_data = create_dataframe(test_path)

    if verbose:
        print('Train data shape: {}'.format(train_data.shape))
        print('Test data shape: {}'.format(test_data.shape))

    train_data_len = train_data.shape[0]
    combined_data = combine_data(train_data, test_data)

    if exclude_columns:
        combined_data = drop_columns(combined_data, exclude_columns)

    format_sex_column(combined_data)
    combined_data = add_title_column(combined_data)

    train_data, test_data = split_dataframe(combined_data, train_data_len)

    if verbose:
        print('\nTrain data shape: {}'.format(train_data.shape))
        print('Test data shape: {}'.format(test_data.shape))

    return train_data, test_data
