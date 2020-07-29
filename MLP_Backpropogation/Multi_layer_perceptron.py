##Imported Libraries
import random                                   ## For setting random state
import math                                     ## For activation Function
import pandas as pd
import numpy as np

##Read and Print Dataset
Data = pd.read_csv('train_data.csv')

##Read and Print Labels
label = pd.read_csv('train_labels.csv')

## Splitting of Whole-Data into Train and Test(Validation) dataset 
train_data= Data.iloc[0:19000,:]
train_label= label.iloc[0:19000,:]
validation_data= Data.iloc[19000:,:]
validation_label= label.iloc[19000:,:]

tr_data =  np.asarray(train_data).T
tr_label = np.asarray(train_label).T
val_data = np.asarray(validation_data).T
val_label = np.asarray(validation_label).T

print("-------------------------------")
print("Training Data:")
print(tr_data)
print("-------------------------------")
print("Training label:")
print(tr_label)
print("-------------------------------")
print("Validation Data:")
print(val_data)
print("-------------------------------")
print("Validation label:")
print(val_label)
print("-------------------------------")

##Random Initialization of Weights and Bias
def random_initiate(IN, HN, ON):                                                    ## (IN: Inout layer neurons, HN: Hidden layer neurons, ON: Output layer neurons)

    ## Initial Weights and bias between Hidden and Input layer
    HI_weight = np.random.randn(HN, IN) * 0.01                        ## Weight matrix of size (Hidden neuron, Input neuron)
    HI_bias = np.zeros(shape=(HN, 1))
    
    ## Initial Weights and bias between Output and Hidden layer
    OH_weight = np.random.randn(ON, HN) * 0.01                         ## ## Weight matrix of size (Output neuron, Hidden neuron)
    OH_bias = np.zeros(shape=(ON, 1))
    
    ## Dictionary which will store all the weights and bias throughtout the MLP.
    weight_bias_dict = {
        "Weight1": HI_weight,
        "bias1": HI_bias,
        "Weight2": OH_weight,
        "bias2": OH_bias
    }
    # print(weight_bias_dict)
    return weight_bias_dict

##Forward Propogation 
def forward_propogation(previous_input, weight, bias, activation):
    '''
    During Forward Propogation, particular neuron performs 2 operations:
    (i) Weighted Sum of Inputs and corresponding Weights.
    (ii) Pass the Weighted Sum to an Activation Function to produce neuron output.
    (iii) Initial values are stored, which are needed as input during backpropogation.

    '''
    
    if activation == "sigmoid":
        '''
        Sigmoid is used for Hidden layer since we can easil take derivative of Sigmoid function which is needed for backpropogation.

        '''
        sum_output,c = Summation(previous_input, weight, bias)
        neuron_output = sigmoid(sum_output)
         
    elif activation == "softmax":
        '''
        Softmax is used for Output layer as it works good for one hot encoded labels.
        
        '''
        sum_output,c = Summation(previous_input, weight, bias)
        neuron_output = softmax(sum_output)
        
    act_c = sum_output
    total_cache = (c, act_c)        

    return neuron_output, total_cache

##Neuron Operations (Summation and Activation Function)
def Summation(each_input, each_weight, bias):
    '''
    Summation output = Multiply each input with its corresponding weight, sum it up and then bias.

    '''
    sum_output = np.dot(each_weight,each_input) + bias
    argument_cache = (each_input, each_weight, bias) 

    return sum_output, argument_cache

def sigmoid(value):
    sgm = 1.0 / (1.0 + np.exp(-value))
    return sgm

def softmax(value):
    sfm = np.exp(value - value.max())
    return sfm / np.sum(sfm, axis=0)

##Error Calculation
def find_error(O_label, A_label):
    '''
    - O_label: Output Labels
    - A_label: Actual Labels
    - Cross Entropy Error function has been used

    '''
    z = A_label.shape[1]
    err = (-1 / z) * np.sum(np.multiply(A_label, np.log(O_label)) + np.multiply(1 - A_label, np.log(1 - O_label)))
    err = np.squeeze(err)
    #print(err)
    return err

##BackPropogation
def backpropagation(drv_activated_output, C, activation_function):  

    if activation_function == "sigmoid":
        backward_neuron_output = sigmoid_derivative(drv_activated_output, C[1])
        
    elif activation_function == "softmax":
        backward_neuron_output = drv_activated_output
        
    drv_activated_output_prev, drv_weight, drv_bias = backward_propogation(backward_neuron_output, C[0])
    
    return drv_activated_output_prev, drv_weight, drv_bias   ## drv_weight: Gradient Weight, drv_bias: Gradient Bias


##Neuron Operation during backpropagation
def sigmoid_derivative(drv_activated_output, value):
    output = 1/(1+np.exp(-value))
    derv_sig_output = drv_activated_output * output * (1-output) 

    return derv_sig_output  ## backward_neuron_output

def backward_propogation(backward_neuron_output, cache):  
    n = cache[0].shape[1]
    drv_each_weight = np.dot(backward_neuron_output, cache[0].T)
    drv_weight = drv_each_weight / n
    drv_each_bias = np.sum(backward_neuron_output, axis=1, keepdims=True)
    drv_bias = drv_each_bias / n
    drv_activated_output_prev = np.dot(cache[1].T, backward_neuron_output)
      
    return drv_activated_output_prev, drv_weight, drv_bias


##Update Weight and Bias Process
def weight_bias_update(WB_dict, gradient, lr): 
    num_layer = len(WB_dict) // 2 
    for l in range(num_layer):
        WB_dict["Weight" + str(l + 1)] = WB_dict["Weight" + str(l + 1)] - lr * gradient["drv_weight" + str(l + 1)]
        WB_dict["bias" + str(l + 1)] = WB_dict["bias" + str(l + 1)] - lr * gradient["drv_bias" + str(l + 1)]
        
    return WB_dict

##MLP Model
def MLP_model(training_data, training_label, neurons_in_layer, lr=0.1, epochs = 160):
    (input_neuron, hidden_neuron, output_neuron) = neurons_in_layer
    np.random.seed(1)  ## Setting Seed for Training dataset
    n_errors = []
    lr = 0.1

    ## Initialize random weights and bias of MLP
    random_weight_bias = random_initiate(input_neuron, hidden_neuron, output_neuron)
    
    ## Random Weights and Bias for forward propogation 
    HI_weight = random_weight_bias["Weight1"]
    HI_bias = random_weight_bias["bias1"]
    OH_weight = random_weight_bias["Weight2"]
    OH_bias = random_weight_bias["bias2"]
    
    # (Forward + Backpropogation + Update weights and bias) for each epoch
    gradient = {}
    for i in range(0, epochs):
       
        # Forward propagation between input and hidden layer
        HI_activated_output, C1 = forward_propogation(training_data, HI_weight, HI_bias,"sigmoid")
        
        # Forward propagation between hidden and output layer
        OH_activated_output, C2 = forward_propogation(HI_activated_output, OH_weight, OH_bias,"softmax")

        
        # Calculation Error/cost function
        error = find_error(OH_activated_output, training_label)
        n_errors.append(error)
        print("Error at epoch %d" %i, error)
   
        
        # Initializing backward propagation
        drv_O_activated_output = - (np.divide(training_label, OH_activated_output) - np.divide(1 - training_label, 1 - OH_activated_output))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        drv_OH_activated_output, drv_weight2, drv_bias2 = backpropagation(drv_O_activated_output, C2,"softmax")
        drv_HI_activated_output, drv_weight1, drv_bias1 = backpropagation(drv_OH_activated_output, C1,"sigmoid")  ## (WARNING: drv_HI_activated_output is not used)
    
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        gradient['drv_weight1'] = drv_weight1
        gradient['drv_bias1'] = drv_bias1
        gradient['drv_weight2'] = drv_weight2
        gradient['drv_bias2'] = drv_bias2
        
        # Update parameters.
        updated_parameters = weight_bias_update(random_weight_bias, gradient, lr)
        #print(parameters)
   
        # Reclaim Weight and Bias from parameters after updating.
        HI_weight = updated_parameters["Weight1"]
        HI_bias = updated_parameters["bias1"]
        OH_weight = updated_parameters["Weight2"]
        OH_bias = updated_parameters["bias2"]

    return updated_parameters

##MLP Layered Architecture
#Neurons in each layer
input_neuron = 784
hidden_neuron = 10
output_neuron = 4
NN = (input_neuron, hidden_neuron, output_neuron)

### Calling MLP with defined parameters
## Computation of Predicted Labels
# Forward Propogation with Inputs obtained after training MLP to obtain "Predicted Labels"
def test_forward_propogate(new_parameters, val_data):
    ##Forward Propogation with New weights
    new_HI_weight = new_parameters["Weight1"]
    new_HI_bias = new_parameters["bias1"]
    new_OH_weight = new_parameters["Weight2"]
    new_OH_bias = new_parameters["bias2"]

    #Forward propogation between Input and Hidden layer
    new_HI_activation_output = forward_propogation(val_data, new_HI_weight, new_HI_bias,"sigmoid")

    #Forward propogation between Input and Hidden layer
    new_OH_activation_output= forward_propogation(new_HI_activation_output[0], new_OH_weight, new_OH_bias,"softmax")

    ## Predicted Labels
    predicted_labels = np.argmax(new_OH_activation_output[0].T,axis=1)

    return predicted_labels   ## Numpy Array (as required)

## Output Labels
output_labels = np.argmax(val_label.T, axis =1)
# print(output_labels)

## Accuracy Function (as defined in provided template on learn)
def accuracy(y_true, y_pred):
    if not (len(y_true) == len(y_pred)):
        print('Size of predicted and true labels not equal.')
        return 0.0

    corr = 0
    for i in range(0,len(y_true)):
        corr += 1 if (y_true[i] == y_pred[i]).all() else 0

    return corr/len(y_true)*100

## Final Accuracy

'''   
After running on defined parameters, I was able to Validation accuracy = 94.42%

'''

## Save Model using Pickle
import pickle

'''
MLP Model will be saved in "MLP_Model.pkl" file

'''



if __name__=='__main__':
    new_parameters = MLP_model(tr_data, tr_label, neurons_in_layer = NN)
   
    ##Predicted Labels
    predicted_labels = test_forward_propogate(new_parameters, val_data) 
    # print(predicted_labels)

    Accuracy = accuracy(output_labels,predicted_labels)
    print("------------------------------------------------------------------")
    print("Validation Accuracy : ", Accuracy,"% with following parameters:")
    print("1. Input Layer Neurons = %d" %input_neuron)
    print("2. Hidden Layer Neurons = %d" %hidden_neuron)
    print("3. Output Layer Neurons = %d" %output_neuron)
    print("4. Learning rate = 0.1")
    print("5. Number of Epochs = 160")
    print("------------------------------------------------------------------")
        

    picklefile = open("Saved_Model.pkl", 'wb')
    pickle.dump(new_parameters, picklefile)
    picklefile.close()
