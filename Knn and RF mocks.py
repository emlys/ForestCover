 from sklearn.ensemble import ExtraTreesClassifier as XTC
#assumes existence of NN named model

#flatten the output of given NN, feed in to KNN
#maybe just use Flatten() from CNN?
flat_dim_size = np.prod(model.output_shape[1:])
knn_input  = Reshape(target_shape=(flat_dim_size,))(model.output)

#knn is linear activations
#
knn_model  = Dense(	units= model. ,
         		   	activation='linear',
          			use_bias=False)(In)

    classifier = Model(inputs=[model.input], outputs=x)
    return classifier


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

 # ONSOMBULL GANG, ExTrA TrEeS gAnG (extra randomForest )
 classifier = XTC(	n_estimators = 					#numba of trees, when the weak band together they are STRONK (or accurate idc)
 					min_samples_split = 			#Node splitting threshold (too many, get out)
 					n_jobs = -1     				# -1 is ALL SYSTEMS GO, anything else is just how many jobs to run para
 					random_state= 					#no fuggin cllue m8
 					verbose = 1						#set to 0 or delete if talks too much
 					warm_start = true				#use last soln, then add more trees(fuck it) 
 					# max_samples =					#i think this can be used for minibatching the trees

 					)