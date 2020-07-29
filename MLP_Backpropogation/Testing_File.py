import numpy as np
import pandas as pd
import pickle
from Multi_layer_perceptron import forward_propogation
STUDENT_NAME = 'AKSHAY GOYAL'
STUDENT_ID = '20842555'

## Required Template
def test_mlp(data_file):
	# Load the test set
	data = pd.read_csv(data_file)
	test_data = np.asarray(data.T)  
	## ***********(PLEASE USE "pd.read_csv(data_file, header = None)" if you have header in your test file.*******

	## Open Pickle File
	P_file = open('Saved_Model.pkl', 'rb')
	new_parameters = pickle.load(P_file)

	## Retrieveing Updated Weight and Bias from already trained model: Model_MLP.py  
	new_HI_weight = new_parameters["Weight1"]
	new_HI_bias = new_parameters["bias1"]
	new_OH_weight = new_parameters["Weight2"]
	new_OH_bias = new_parameters["bias2"]

	## Forward propogation with updated weights and bias along with test data
	new_HI_activation_output = forward_propogation(test_data, new_HI_weight, new_HI_bias,"sigmoid")  					## FOr Hidden and input layer
	new_OH_activation_output= forward_propogation(new_HI_activation_output[0], new_OH_weight, new_OH_bias,"softmax")  	## For output and hidden layer
	predicted_labels = np.argmax(new_OH_activation_output[0].T,axis=1)
 
	## Convert Predicted labels to One Hot Encoded Labels
	y_pred = []
	for value in predicted_labels:
		label = np.zeros(4)
		label[value] = 1
		y_pred.append(label)
	y_pred = np.asarray(y_pred)
	print("Predicted Labels Type: ", type(y_pred)) ## NumpyArray as required
	
	return y_pred 

## *************************************DISCLAIMER******************************************************************************
## (Below is the template of how I tested my code and this template is exactly similar to what was provided on learn,
## so I hope you wont find any issues while testing the model. If faced any, kindly contact a49goyal@uwaterloo.ca)

'''
from test_mlp import test_mlp, STUDENT_NAME, STUDENT ID
from acc_calc import accuracy 
import pandas as pd
import numpy as np

y_pred = test_mlp('sample_new.csv')
print("Predicted Labels type:\n", type(y_pred))

test_labels = np.asarray("test_labels") ## (Instructor will provide testing labels as one hot encoded in numpy array)

test_accuracy = accuracy(test_labels, y_pred) * 100
print("Final Accuracy on Testing Data: ", test_accuracy, "%")

'''