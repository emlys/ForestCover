import pandas
import math
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier as XTC


# this part is just straight-up Emily's code

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

# this part is (roughly) from assignment 2
'''
# set up the neural network
input_layer = Input(shape=(55,))

# add dropout here somehow
x0 = Dense(200, activation='relu', input_shape=(55,))(input_layer)
x1 = Dense(100, activation='relu')(x0)
x2 = Dense(50, activation='relu')(x1)
x3 = Dense(30, activation='relu')(x2)
x4 = Dense(15, activation='softmax')(x3)

output_layer = Dense(7, activation='sigmoid')(x4)

mod = Model(inputs=input_layer, outputs=output_layer)

print("network made, now compiling")
# compile the neural network
mod.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
'''
 

#Extremely Randomized Trees
classifier = XTC(	
					n_estimators = 200,					# Number of weak learners in the forest (Decision Trees)
					max_features=  "sqrt",				# Max number of features to consider while attempting classification
					min_samples_split =  2,				# Minimum number of features required to make further splits, # < 2 is a leaf.
					n_jobs = -1,						# Run as many parallel jobs as possible (for speed)
					random_state= 5,					# Random for Sampling of features @bestsplit (bootstrap,draw for each maxfeature too but this is more imporant)
					verbose = 0,						# set to 0 or delete if talks too much
					warm_start = True,					# use previous solution, and add more trees.
					#bootstrap = True  					# MiniBatch the trees, sampling from data set
					#max_samples = 2000					# Bootstrapping/minibatching, but we split our data ahead of time anyways
				)

print('compiled, now running')
# train the neural network
#history = mod.fit(x_train, y_train, 32, 300, validation_split=0.2)
fit = classifier.fit(x_train, y_train)

print("running score func")

acc = classifier.score(x_test, y_test)

print(acc)

'''
# evaluate for success
train_loss, train_acc = mod.evaluate(x_train, y_train)
test_loss, test_acc = mod.evaluate(x_test, y_test)

print("Training set accuracy: ", train_acc)
print("Test set accuracy: ", test_acc)


'''
