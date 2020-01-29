import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras


class ForestCover:

    # read in the data into dataframes
    data = pandas.read_csv('train.csv')

    # split into x (the features) and y (the target information)
    # x has all columns except 'Cover_Type'
    # despite the name, this does not alter the original df
    x = data.drop('Cover_Type', axis=1).values

    # y has just the column 'Cover_Type'
    # use sklearn's LabelBinarizer to convert the one column 
    # with labels 1-7 into 7 one-hot encoded columns
    classes = list(range(1, 8))
    y = label_binarize(data[['Cover_Type']], classes)
    print(type(y))

    # split into train and test sets, 20% of data set aside for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(type(x_train), type(y_test))
    print(x.shape, y.shape)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(55,), activation='relu'),
        keras.layers.Dense(7, activation='sigmoid')])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)
