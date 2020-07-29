import numpy as np
import pandas as pd
import pickle
from Multi_layer_perceptron import forward_propogation

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
	print("Predicted Labels Type: ", type(y_pred))
	return y_pred