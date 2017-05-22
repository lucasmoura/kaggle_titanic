import pandas
import pickle
import random

import numpy as np


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
    dataframe['Title'] = dataframe.Title.replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
         'Jonkheer', 'Dona'], 'Rare')

    dataframe['Title'] = dataframe.Title.replace('Mlle', 'Miss')
    dataframe['Title'] = dataframe.Title.replace('Ms', 'Miss')
    dataframe['Title'] = dataframe.Title.replace('Mme', 'Mrs')

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataframe['Title'] = dataframe['Title'].map(title_mapping)
    dataframe['Title'] = dataframe['Title'].fillna(0)

    return dataframe


def add_isalone_column(dataframe):
    dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch'] + 1
    dataframe['IsAlone'] = 0

    dataframe.loc[dataframe['FamilySize'] == 1, 'IsAlone'] = 1
    dataframe = drop_columns(dataframe, ['FamilySize', 'SibSp', 'Parch'])

    return dataframe


def add_ageclass_column(dataframe):
    dataframe['AgeClass'] = dataframe['Age'] * dataframe['Pclass']
    return dataframe


def create_new_columns(dataframe):
    dataframe = add_title_column(dataframe)
    dataframe = add_isalone_column(dataframe)
    dataframe = add_ageclass_column(dataframe)
    return dataframe


def split_dataframe(dataframe, split_dataframe_len):
    return dataframe[:split_dataframe_len], dataframe[split_dataframe_len:]


def format_sex_column(dataframe):
    """
    The Sex column in the dataframe has the as values the actual gender
    name of a passenger. In order to use that information, it will be
    required to change these values to numerical ones:

    0 = female
    1 = male
    """
    dataframe.loc[dataframe.Sex == 'female', 'Sex'] = 0
    dataframe.loc[dataframe.Sex == 'male', 'Sex'] = 1


def format_embarked_column(dataframe):
    """
    The Embarked column in the dataframe has three possible values for where
    the passenger has embarked. This locations are represented by a single
    uppercase char caracter. This function will replace that char
    representation to a numeric one:

    C = 0
    Q = 1
    S = 2
    """
    dataframe.loc[dataframe.Embarked == 'C', 'Embarked'] = 0
    dataframe.loc[dataframe.Embarked == 'Q', 'Embarked'] = 1
    dataframe.loc[dataframe.Embarked == 'S', 'Embarked'] = 2


def format_fare_column(dataframe):
    dataframe.loc[dataframe['Fare'] <= 7.91, 'Fare'] = 0
    dataframe.loc[(dataframe['Fare'] > 7.91) & (dataframe['Fare'] <= 14.454),
                  'Fare'] = 1
    dataframe.loc[(dataframe['Fare'] > 14.454) & (dataframe['Fare'] <= 31),
                  'Fare'] = 2
    dataframe.loc[dataframe['Fare'] > 31, 'Fare'] = 3

    dataframe['Fare'] = dataframe['Fare'].astype(int)


def format_age_column(dataframe):
    dataframe.loc[dataframe['Age'] <= 16, 'Age'] = 1
    dataframe.loc[(dataframe['Age'] > 16) & (dataframe['Age'] <= 32),
                  'Age'] = 2
    dataframe.loc[(dataframe['Age'] > 32) & (dataframe['Age'] <= 48),
                  'Age'] = 3
    dataframe.loc[(dataframe['Age'] > 48) & (dataframe['Age'] <= 64),
                  'Age'] = 4
    dataframe.loc[dataframe['Age'] > 64, 'Age'] = 5


def format_column_values(dataframe):
    format_sex_column(dataframe)
    format_embarked_column(dataframe)
    format_fare_column(dataframe)
    format_age_column(dataframe)


def fill_missing_ages(dataframe, verbose=False):
    """
    This function is used to guess ages for passengers with missing ages.
    To guess an age, we look at two features, the Pclass and Sex. Every
    combination of Pclass and Sex will have a fit random age to replace a
    missing one.
    """
    guess_ages = np.zeros((2, 3))

    print('Filling age missing values...')

    for i in range(2):  # Sex
        for j in range(3):  # Pclass
            sex = 'female' if i == 0 else 'male'
            guess_age = dataframe[((dataframe['Sex'] == sex) &
                                   (dataframe['Pclass'] == j + 1))]
            guess_age = guess_age['Age'].dropna()

            guess_ages[i, j] = guess_age.mean()

            if verbose:
                print('Guessed age for Sex {} and Pclass {}: {}'.format(
                    i, j+1, int(guess_ages[i, j])))

    print()
    for i in range(2):
        for j in range(3):
            sex = 'female' if i == 0 else 'male'
            dataframe.loc[
                ((dataframe.Age.isnull()) & (dataframe.Sex == sex) &
                 (dataframe.Pclass == j + 1)), 'Age'] = guess_ages[i, j]

    dataframe['Age'] = dataframe['Age'].astype(int)


def fill_missing_embarked(dataframe, verbose=False):
    """
    Fill the missing Embarked values with the most frequent embarked location.
    """
    most_frequent_embarked = dataframe.Embarked.dropna().mode()[0]
    dataframe.Embarked = dataframe.Embarked.fillna(most_frequent_embarked)


def fill_missing_fares(dataframe, verbose=False):
    dataframe['Fare'].fillna(dataframe['Fare'].dropna().median(), inplace=True)


def fill_missing_values(dataframe, verbose):
    fill_missing_ages(dataframe, verbose)
    fill_missing_embarked(dataframe, verbose)
    fill_missing_fares(dataframe, verbose)


def create_create_new_columns(dataframe):
    dataframe = add_title_column(dataframe)
    return dataframe


def print_overall_info(dataframe):
    print('Displaying general info...')
    print(dataframe.info(memory_usage=False))
    print()


def print_data_information(dataframe, name):
    print('{} data shape: {}'.format(name, dataframe.shape))

    missing_values = False
    for value in dataframe.columns.values:
        nan_count = dataframe[value].isnull().sum()
        if nan_count:
            missing_values = True
            print('Number of missing values for {}: {}'.format(
                value, nan_count))

    if not missing_values:
        print('No missing values for {} data'.format(name))

    print()


def format_data(train_path, test_path, exclude_columns=None, verbose=False):
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
    train_data = create_dataframe(train_path)
    test_data = create_dataframe(test_path)

    if verbose:
        print('Before data formatting...\n')
        print_overall_info(train_data)
        print_data_information(train_data, 'Train')
        print_data_information(test_data, 'Test')

    train_data_len = train_data.shape[0]
    combined_data = combine_data(train_data, test_data)

    fill_missing_values(combined_data, verbose)
    format_column_values(combined_data)
    combined_data = create_new_columns(combined_data)
    combined_data = exclude_columns_from_data(combined_data, exclude_columns)

    train_data, test_data = split_dataframe(combined_data, train_data_len)
    train_data = drop_columns(train_data, ['PassengerId'])

    if verbose:
        print('After data formatting...\n')
        print_data_information(train_data, 'Train')
        print_data_information(test_data, 'Test')

    return train_data, test_data


def create_data_array(data):
    data = drop_columns(data, ['Survived']).as_matrix()
    num_features = data.shape[1]

    return [np.reshape(x, (num_features, 1)) for x in data]


def format_training_data(train_data, single=False):
    """
    This method will be used to format the training data. Every passenger data
    will be formatted to the following pattern:

    (x, y)

    where x will be the passenger's features and y will if that passsenger
    survived or not.
    """
    survived = train_data.Survived.as_matrix().astype(int)

    if single:
        y = create_single_output(survived)
    else:
        y = create_two_output(survived)

    train_data = create_data_array(train_data)

    return list(zip(train_data, y))


def create_single_output(output):
    return [np.array(pred).reshape((1, 1)) for pred in output]


def create_two_output(output):
    y = np.zeros((output.shape[0], 2))

    for index, value in enumerate(output.tolist()):
        y[index][value] = 1

    y = y.astype(int)
    return [np.reshape(x, (2, 1)) for x in y]


def create_validation_data(train_data, size=0.1, verbose=False):
    """
    This method will split the train data into train and validation data based
    on the parameter size, which should the proportion of training data that
    will be transformed into validation data.
    """
    random.shuffle(train_data)
    num_validation = int(len(train_data) * size)

    validation_data = train_data[:num_validation]
    train_data = train_data[num_validation:]

    if verbose:
        print('Size of training set after split: {}'.format(
            len(train_data)))
        print('Size of validation set after split: {}'.format(
            len(validation_data)))

    return (train_data, validation_data)


def save_data(data, file_name):
    with open(file_name, 'wb') as data_file:
        pickle.dump(data, data_file)


def load_data(data_path):
    with open(data_path, 'rb') as data_file:
        return pickle.load(data_file)
