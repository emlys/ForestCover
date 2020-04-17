import itertools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import label_binarize, MinMaxScaler, PowerTransformer
import tensorflow as tf
from tensorflow import keras
import pprint
import sys

from test import Test

class ForestCover:

    def __init__(self):
        # read in the data into dataframes
        data = pd.read_csv('train.csv')

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

    def run(self):

        # Define the model architecture
        model = keras.Sequential([
            # input size is fixed at 55 because there are 55 features
            keras.layers.Dense(128, input_dim=55, activation='sigmoid'),
            keras.layers.Dense(64, activation='sigmoid'),
            # output size is fixed at 7 because there are 7 possible classes
            # output layer activation is fixed at softmax because this is recommended 
            # for multiclass classification problems
            keras.layers.Dense(7, activation='softmax')])

        # Compile the model with desired optimizer and loss
        model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.05),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        # Fit the model 5 times using 5-fold cross validation
        for train_index, test_index in KFold(5, shuffle=True).split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model.fit(x_train, y_train, epochs=10, batch_size=128)
            y_pred = model.predict_classes(x_test, verbose=2)

        # Convert the one-hot target values back to scalars so that they can be input to the confusion matrix
        y_true = np.array([np.argmax(y) for y in y_test])
        self.plot_confusion_matrix(y_true, y_pred)

    def run_kfold_crossvalidation(self, model, optimizer, learning_rate, epochs, batch_size, k=5):

        losses, accuracies = [], []
        
        # Fit the model k times using k-fold cross validation
        for train_index, test_index in KFold(k, shuffle=True).split(self.x):
            # Get the train and test sets for this fold
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Model needs to be re-compiled for each fold so that we are starting over fresh
            model = self.compile_model(model, optimizer, learning_rate)

            # Train the model on the training data for this fold
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

            # Evaluate the model on the testing data for this fold
            loss, accuracy = tuple(model.evaluate(x_test, y_test, batch_size=batch_size))
            losses.append(loss)
            accuracies.append(accuracy)

        # Return a dictionary with average loss and accuracy over all the folds
        return (sum(losses) / k, sum(accuracies) / k)

    def compile_model(self, model: keras.Sequential, optimizer, learning_rate: float):
        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def plot_confusion_matrix(self, y_true, y_pred):
        print(y_true, y_pred)
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize='true')

        classes  = [self.class_names[i] for i in range(1, 8)]
        
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        
        
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

        # add labels
        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label', rotation_mode='anchor')

        # Format plot
        plt.colorbar()
        plt.tight_layout()
        plt.gcf().set_size_inches(6, 6)

        plt.show()

  

if __name__ == '__main__':
    fc = ForestCover()

    # Define the model architectures we want to test
    models = [
    keras.Sequential([
        keras.layers.Dense(128, input_dim=55, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(7, activation='softmax')]),
    keras.Sequential([
        keras.layers.Dense(128, input_dim=55, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(7, activation='softmax')]),
    keras.Sequential([
        keras.layers.Dense(256, input_dim=55, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(32, activation='sigmoid'),
        keras.layers.Dense(7, activation='softmax')]),
    # keras.Sequential([
    #     keras.layers.Dense(256, input_dim=55, activation='sigmoid'),
    #     keras.layers.Dense(512, activation='sigmoid'),
    #     keras.layers.Dense(256, activation='sigmoid'),
    #     keras.layers.Dense(128, activation='sigmoid'),
    #     keras.layers.Dense(64, activation='sigmoid'),
    #     keras.layers.Dense(32, activation='sigmoid'),
    #     keras.layers.Dense(7, activation='softmax')])
    ]

    # Different model compilation options to try
    optimizers = [keras.optimizers.SGD]
    learning_rates = [.05, .001, .0005]

    # Different training options to try
    epochs = [10]
    batch_size = [1, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192]

    # Generate all combinations of the compilation options
    compile_options = [{'optimizer': combo[0], 'learning_rate': combo[1]} 
        for combo in Test.generate_all_combinations([optimizers, learning_rates])]

    # Generate all combinations of the training options
    train_options = [{'epochs': combo[0], 'batch_size': combo[1]} 
        for combo in Test.generate_all_combinations([epochs, batch_size])]

    results = pd.DataFrame(columns=['model', 'optimizer', 'learning_rate', 'epochs', 'batch_size', 'loss', 'accuracy'])

    # For each model architecture
    for i, model in enumerate(models):
        print("Model " + str(i))

        # For each combo of compilation options
        for j, c in enumerate(compile_options):

            print("Compiling option " + str(j))

            # For each combo of training options
            for k, t in enumerate(train_options):
                print("Training option " + str(k))

                # Get average loss and accuracy over 5-fold cross validation training
                loss, accuracy = fc.run_kfold_crossvalidation(
                    model=model,
                    optimizer=c['optimizer'],
                    learning_rate=c['learning_rate'],
                    epochs=t['epochs'],
                    batch_size=t['batch_size']
                )

                # Add the data as a new row to the results dataframe
                results = results.append({
                    'model': i,
                    'optimizer': c['optimizer'],
                    'learning_rate': c['learning_rate'],
                    'epochs': t['epochs'],
                    'batch_size': t['batch_size'],
                    'loss': loss,
                    'accuracy': accuracy
                }, ignore_index=True)
    results.to_csv('out.csv')
                


