import itertools
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
import pprint
import sys


class ForestCover:

    def __init__(self):
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

        # split into train and test sets, 20% of data set aside for testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)
        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    def run(self):
        model = keras.Sequential([
            # input size is fixed at 55 because there are 55 features
            keras.layers.Dense(64, input_dim=55, activation='sigmoid'),
            # output size is fixed at 7 because there are 7 possible classes
            # output layer activation is fixed at softmax because this is recommended 
            # for multiclass classification problems
            keras.layers.Dense(7, activation='softmax')])

        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=5)
        model.evaluate(self.x_test, self.y_test, verbose=2)

    def create_model(self, size, optimizer, learning_rate, loss, hidden_activation):
        """Return a keras model with given params

        size: number of nodes in the hidden layer (just using 1 hidden layer for now)
        optimizer: keras optimizer class e.g. keras.optimizers.SGD
        loss: keras loss function name
        hidden_activation: keras activation function name to use for the hidden layer
        """
        model = keras.Sequential([
            # input size is fixed at 55 because there are 55 features
            keras.layers.Dense(size, input_dim=55, activation=hidden_activation),
            # output size is fixed at 7 because there are 7 possible classes
            # output layer activation is fixed at softmax because this is recommended 
            # for multiclass classification problems
            keras.layers.Dense(7, activation='softmax')])

        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy'])

        return model

    def test(self):

        # this will store a dataframe of results concatenated from each iteration
        results = None

        # define all architecture options to combine
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        optimizers = [keras.optimizers.SGD]
        learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
        losses = ['categorical_crossentropy']
        hidden_activations = ['relu', 'sigmoid', 'tanh']

        # generate a list of all possible combinations of the architecture options
        options = [sizes, optimizers, learning_rates, losses, hidden_activations]
        combos = list(itertools.product(*options))

        # define all hyperparameter options to try in grid search
        param_grid = {'epochs': [4], 'batch_size': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]}

        # for each combo of architecture options, do grid search over all the hyperparameter options
        for i, c in enumerate(combos):
            params = {
                'size': c[0],
                'optimizer': c[1],
                'learning_rate': c[2],
                'loss': c[3],
                'hidden_activation': c[4]}

            print("testing architecture param combo {} out of {}:".format(i + 1, len(combos)))
            print(params)
      
            # build and compile the model using the architecture options for this iteration
            # this wrapper for the keras model allows it to be used with sklearn GridSearchCV
            keras_model = keras.wrappers.scikit_learn.KerasClassifier(
                build_fn=ForestCover().create_model,
                size=params['size'],
                optimizer=params['optimizer'],
                learning_rate=params['learning_rate'],
                loss=params['loss'],
                hidden_activation=params['hidden_activation'])

            self.grid_search = sklearn.model_selection.GridSearchCV(keras_model, param_grid)

            # this does 5-fold cross validation by default
            # validation data is set to self.x_test, self.y_test so that it is the same across all 
            # GridSearchCV instances used in the tests
            self.grid_search.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), verbose=0)

            if results is None:
                results = self.merge_params(params, self.grid_search.cv_results_)
            else:
                results = pandas.concat([results, self.merge_params(params, self.grid_search.cv_results_)])
            
        self.output(results)

    def output(self, results):
        col_order = ['size', 'optimizer', 'loss', 'hidden_activation', 'param_epochs', 'param_batch_size', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'mean_score_time']
        results[col_order].to_csv("out.csv")

    def merge_params(self, architecture_params, gridsearch_params):
        """Return a pandas DataFrame with both architecture and grid search params for each row

        architecture_params: dict describing the architecture params applied to all rows of the gridsearch,
            e.g. {'size': s, 'optimizer': o, 'loss': l, 'hidden_activation': h}
        gridsearch_params: GridSearchCV.cv_results_ describing grid search params and scores for each combo
            e.g. {key: [list of length n]} where n = number of different combos
        """

        # get the number of rows in the grid (different combos of grid search params)
        row_count = len(gridsearch_params['mean_test_score'])

        # expand each architecture param value into a list,
        # to show that it applies to each row in the gridsearch
        expanded_params = {key: [val for i in range(row_count)] for key, val in architecture_params.items()}

        # add the items from expanded_params into the gridsearch dictionary
        gridsearch_params.update(expanded_params)

        # export to a DataFrame object
        return pandas.DataFrame.from_dict(gridsearch_params)
  

if __name__ == '__main__':
    ForestCover().test()

