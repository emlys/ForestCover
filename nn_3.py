import pandas
import math
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier as ERT

# read in the data
print("reading in data!")
data = pandas.read_csv('train.csv')

# Split into x (features) and y (target info)
x = data.drop('Cover_Type', axis=1).values

classes = list(range(1,8))
y = label_binarize(data[['Cover_Type']], classes)

# set aside a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)
print("been split, setting up network now")

 

#Extremely Randomized Trees
classifier = ERT(	
					n_estimators = 200,					# Number of weak learners in the forest (Decision Trees)
					max_features =  "sqrt",				# Max number of features to consider while attempting classification
					min_samples_split =  2,				# Minimum number of features required to make further splits, # < 2 is a leaf.
					n_jobs = -1,						# Run as many parallel jobs as possible (for speed)
					random_state = 5,					# Random for Sampling of features @bestsplit (bootstrap,draw for each maxfeature too but this is more imporant)
					verbose = 0,						# set to 0 or delete if talks too much
					warm_start = True,					# use previous solution, and add more trees.
					#bootstrap = True  					# MiniBatch the trees, sampling from data set
					#max_samples = 2048					# Bootstrapping/minibatching, but we split our data ahead of time anyways
				)

print('compiled, now running')

# Fitting
fit = classifier.fit(x_train, y_train)

print("running score function")
acc = classifier.score(x_test, y_test)
print("Accuracy: ", acc)

