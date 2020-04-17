import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier as ERT
from sklearn.metrics import confusion_matrix


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
					n_estimators = 100,					# Number of weak learners in the forest (Decision Trees)
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


# Uncomment this section to make a confusion matrix
'''
def plot_confusion_matrix(y, y_pred):
	class_names = {
	    1: 'Spruce/Fir',
	    2: 'Lodgepole Pine',
	    3: 'Ponderosa Pine',
	    4: 'Cottonwood/Willow',
	    5: 'Aspen',
	    6: 'Douglas-fir',
	    7: 'Krummholz'
	}
	# making confusion matrix
	conf_m = confusion_matrix(y, y_pred, normalize='true')
	classes  = [class_names[i] for i in range(1, 8)]
	plt.imshow(conf_m, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix")
	plt.colorbar()
	# display tick marks
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation='vertical')
	plt.yticks(tick_marks, classes)
	for i, j in itertools.product(range(conf_m.shape[0]), range(conf_m.shape[1])):
		plt.text(j, i, "%.2f" % (conf_m[i, j]), 
			horizontalalignment="center", 
			verticalalignment="center", 
			# choose the text color to contrast well against background color
			color="white" if conf_m[i, j] > 0.5 else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label', rotation_mode='anchor')
	plt.gcf().set_size_inches(6, 6)
	plt.show()
	#print(cm(y,predictions))

# Returns the predicted class for each row in one-hot format
predictions = classifier.predict(x_test)
# Convert one-hot formatted predictions into class numbers
predictions = np.array([np.argmax(p) + 1 for p in predictions])
# Convert one-hot formatted true values into class numbers
true_labels = np.array([np.argmax(p) + 1 for p in y_test])
plot_confusion_matrix(true_labels, predictions)

'''
