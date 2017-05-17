import pandas


def create_dataframe(data_path):
    return pandas.read_csv(data_path)


def combine_data(data1, data2):
    return data1.append(data2)


def drop_columns(dataframe, exclude_columns):
    return dataframe.drop(exclude_columns, axis=1)


def exclude_columns_from_data(dataframe, exclude_columns):
    if not exclude_columns:
        return dataframe

    return drop_columns(dataframe, exclude_columns)


def add_title_column(dataframe):
    dataframe['Title'] = dataframe.Name.str.extract(' (\w+)\.', expand=False)
    return dataframe


def split_dataframe(dataframe, split_dataframe_len):
    return dataframe[:split_dataframe_len], dataframe[split_dataframe_len:]


"""
    The Sex column in the dataframe has the as values the actual gender name of
    a passenger. In order to use that information, it will be required to
    change these values to numerical ones:

    0 = female
    1 = male
"""
def format_sex_column(dataframe):
    dataframe.loc[dataframe.Sex == 'female', 'Sex'] = 0
    dataframe.loc[dataframe.Sex == 'male', 'Sex'] = 1


"""
    The Embarked column in the dataframe has three possible values for where
    the passenger has embarked. This locations are represented by a single
    uppercase char caracter. This function will replace that char
    representation to a numeric one:

    C = 0
    Q = 1
    S = 2
"""
def format_embarked_column(dataframe):
    dataframe.loc[dataframe.Embarked == 'C', 'Embarked'] = 0
    dataframe.loc[dataframe.Embarked == 'Q', 'Embarked'] = 1
    dataframe.loc[dataframe.Embarked == 'S', 'Embarked'] = 2


def apply_feature_normalization(dataframe, column_name):
    mean = dataframe[column_name].mean()
    std = dataframe[column_name].std()

    dataframe[column_name] = dataframe[column_name].apply(
        lambda x: (x - mean) / std)


"""
    The Fare column has the amount of money a passenger paid for a ticket. It
    will be better to normalize such a feature, meaning that for every column
    value, we will subtract the value from the mean and divide by the standard
    deviation.
"""
def format_fare_column(dataframe):
    apply_feature_normalization(dataframe, 'Fare')


"""
    The Age column has the real age values of the passengers. We will apply the
    same feature normalization as the one applied for the Fare column.
"""
def format_age_column(dataframe):
    apply_feature_normalization(dataframe, 'Age')


def format_column_values(dataframe):
    format_sex_column(dataframe)
    format_embarked_column(dataframe)
    format_fare_column(dataframe)
    format_age_column(dataframe)


def create_create_new_columns(dataframe):
    dataframe = add_title_column(dataframe)
    return dataframe


def print_overall_info(dataframe):
    print(dataframe.info(memory_usage=False))
    print()


def print_data_information(dataframe, name):
    print('{} data shape: {}'.format(name, dataframe.shape))

    for value in dataframe.columns.values:
        nan_count = dataframe[value].isnull().sum()
        if nan_count:
            print('Number of missing values for {}: {}'.format(
                value, nan_count))

    print()


"""
    Every row on training data:

    * passengerId: The identifier of a given passenger;
    * Survived: If the passenger survived or not (0 = No, 1 = Yes);
    * Pclass: The ticket class for a passenger (1 = 1st, 2 = 2nd, 3 = 3rd);
    * Name: The passenger's name;
    * Sex: The passenger sex (male or female);
    * Age: The age in years of a passenger;
    * SibSp: Number of siblings and spouses aboard;
    * Parch: Number of parents and children aboard;
    * Ticket: The ticker number of the passenger;
    * Fare: The amount of money the passenger has paid for the ticket;
    * Cabin: The passenger's cabin number;
    * Embarked: Where the passenger embarked
                (C = Cherbourgh, Q = Queenstown, S = Southampton)

    This function will them be used to format the data for it to be fit to use
    on a neural network.
"""
def format_data(train_path, test_path, exclude_columns=None, verbose=False):
    train_data = create_dataframe(train_path)
    test_data = create_dataframe(test_path)

    if verbose:
        print_overall_info(train_data)
        print_data_information(train_data, 'Train')
        print_data_information(test_data, 'Test')

    train_data_len = train_data.shape[0]
    combined_data = combine_data(train_data, test_data)

    format_column_values(combined_data)
    fill_missing_values(combined_data)
    # combined_data = create_create_new_columns(combined_data)
    combined_data = exclude_columns_from_data(combined_data, exclude_columns)

    train_data, test_data = split_dataframe(combined_data, train_data_len)
    train_data = drop_columns(train_data, ['PassengerId'])

    if verbose:
        print_data_information(train_data, 'Train')
        print_data_information(test_data, 'Test')

    return train_data, test_data


"""
    This method will be used to format the training data. Every passenger data
    will be formatted to the following pattern:

    (x, y)

    where x will be the passenger's features and y will if that passsenger
    survived or not.
"""
def format_training_data(train_data):
    y = train_data.Survived
    train_data = drop_columns(train_data, ['Survived'])

    return (train_data, y)
