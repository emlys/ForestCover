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
 # ONSOMBULL GANG, ExTrA TrEeS gAnG (extra randomForest )
classifier = XTC(	n_estimators =     200,				# numba of trees, when the weak band together they are STRONK (or accurate idc)
					max_features=  "sqrt",				# max features to consider while attempting classification
					min_samples_split =  2,				# Node splitting threshold (too many, get out)
					n_jobs = -1,						# -1 is ALL SYSTEMS GO, anything else is just how many jobs to run para
					#oob_score = True,
					#bootstrap = True,
					random_state= 5,					# no fuggin cllue m8
					verbose = 0,						# set to 0 or delete if talks too much
					warm_start = True,					# use last soln, then add more trees(fuck it) 
					max_samples = 20					# i think this can be used for minibatching the trees

					)

print('compiled, now running')
# train the neural network
#history = mod.fit(x_train, y_train, 32, 300, validation_split=0.2)
thingy = classifier.fit(x_train, y_train)

print("running score func")

succ = classifier.score(x_test, y_test)

print(succ)

'''
# evaluate for success
train_loss, train_acc = mod.evaluate(x_train, y_train)
test_loss, test_acc = mod.evaluate(x_test, y_test)

print("Training set accuracy: ", train_acc)
print("Test set accuracy: ", test_acc)


'''
