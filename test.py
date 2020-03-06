import itertools
import pandas
import sklearn
from tensorflow import keras

class Test:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test

    def create_model(self, model):
        """Return a keras model compiled with given params

        optimizer: keras optimizer class e.g. keras.optimizers.SGD
        loss: keras loss function name
        learning_rate: keras model learning rate value
        """
        # model = keras.Sequential([
        #     # input size is fixed at 55 because there are 55 features
        #     keras.layers.Dense(128, input_dim=55, activation='sigmoid'),
        #     keras.layers.Dense(64, activation='sigmoid'),
        #     # output size is fixed at 7 because there are 7 possible classes
        #     # output layer activation is fixed at softmax because this is recommended 
        #     # for multiclass classification problems
        #     keras.layers.Dense(7, activation='softmax')])
        modelA = keras.Sequential(model)

        modelA.compile(
            optimizer=keras.optimizers.SGD(), #optimizer(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return modelA

    @staticmethod
    def generate_all_combinations(options: list) -> list:
        """Return a list of all possible combinations of the given options"""
        
        return list(itertools.product(*options))
        

    def test(self, models, param_combos):
        """
        A brute force method to test all combinations of given architecture options and hyperparameters.

        sklearn's GridSearchCV class is handy for comparing hyperparameters and getting performance data,
        but it can't compare multiple architectures. For each combination of given architecture options,
        this method instantiates a GridSearchCV and uses it to test all combinations of the given
        hyperparameters. The GridSearchCV results from each iteration are combined and output to a CSV.
        """

        # this will store a dataframe of results concatenated from each iteration
        results = None

        # define all hyperparameter options to try in grid search
        param_grid = {'epochs': [10], 'batch_size': [128, 256, 512, 1024, 2048, 4096]}

        models = [1]
        
        # for each model, do grid search over all the hyperparameter options
        for i, model in enumerate(models):

            print("testing model {} out of {}:".format(i + 1, len(models)))

            for params in param_combos:

                print("testing param combo {} out of {}:".format(i + 1, len(param_combos)))

                model = [keras.layers.Dense(128, input_dim=55, activation='sigmoid'),
                    keras.layers.Dense(7, activation='softmax')]
            
                # build and compile the model using the architecture options for this iteration
                # this wrapper for the keras model allows it to be used with sklearn GridSearchCV
                keras_model = keras.wrappers.scikit_learn.KerasClassifier(
                    build_fn=self.create_model,
                    model=model
                    # optimizer=keras.optimizers.SGD,
                    # learning_rate=0.001
                )
          
                grid_search = sklearn.model_selection.GridSearchCV(keras_model, param_grid)

                # this does 5-fold cross validation by default
                # validation data is set to self.x_test, self.y_test so that it is the same across all 
                # GridSearchCV instances used in the tests
                grid_search.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), verbose=2)

                if results is None:
                    results = self.merge_params(params, grid_search.cv_results_)
                else:
                    results = pandas.concat([results, self.merge_params(params, grid_search.cv_results_)])
            
        self.output(results)

    def output(self, results):
        col_order = ['optimizer', 'param_epochs', 'param_batch_size', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'mean_score_time']
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



