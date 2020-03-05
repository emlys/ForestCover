import itertools
import numpy as np
import pandas
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import label_binarize, MinMaxScaler, PowerTransformer
import tensorflow as tf
from tensorflow import keras
import pprint
import sys


class ForestCover:

    def __init__(self):
        # read in the data into dataframes
        data = pandas.read_csv('train.csv')

        # forest cover types corresponding to each number
        self.class_names = {
            1: 'Spruce/Fir',
            2: 'Lodgepole Pine',
            3: 'Ponderosa Pine',
            4: 'Cottonwood/Willow',
            5: 'Aspen',
            6: 'Douglas-fir',
            7: 'Krummholz'
        }

        # split into x (the features) and y (the target information)
        # x has all columns except 'Cover_Type'
        # despite the name, 'drop' does not alter the original dataframe
        x = data.drop('Cover_Type', axis=1).values

        # Scale the input data so that each column's [min, max] range is mapped to [0, 1]
        scaler = MinMaxScaler().fit(x)
        self.x = scaler.transform(x)

        # y has just the column 'Cover_Type'
        # use sklearn's LabelBinarizer to convert the one column 
        # with labels 1-7 into 7 one-hot encoded columns
        self.y = label_binarize(data[['Cover_Type']], list(range(1, 8)))

        # split into train and test sets, 20% of data set aside for testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2)

    def run(self):
        model = keras.Sequential([
            # input size is fixed at 55 because there are 55 features
            keras.layers.Dense(128, input_dim=55, activation='sigmoid'),
            keras.layers.Dense(64, activation='sigmoid'),
            # output size is fixed at 7 because there are 7 possible classes
            # output layer activation is fixed at softmax because this is recommended 
            # for multiclass classification problems
            keras.layers.Dense(7, activation='softmax')])

        for train_index, test_index in KFold(5, shuffle=True).split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.05),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5)
            y_pred = model.predict_classes(x_test, verbose=2)

        y_true = np.array([np.argmax(y) for y in y_test])
        self.plot_confusion_matrix(y_true, y_pred)


    def plot_confusion_matrix(self, y_true, y_pred):
        print(y_true, y_pred)
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize='true')

        classes  = [self.class_names[i] for i in range(1, 8)]
        
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()

        # display tick marks
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation='vertical')
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, "%.2f" % (confusion_matrix[i, j]), 
                horizontalalignment="center", 
                verticalalignment="center", 
                # choose the text color to contrast well against background color
                color="white" if confusion_matrix[i, j] > 0.5 else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label', rotation_mode='anchor')
        plt.gcf().set_size_inches(6, 6)
        plt.show()

  

if __name__ == '__main__':
    ForestCover().run()

