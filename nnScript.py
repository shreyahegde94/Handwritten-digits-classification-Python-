import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

selected_features = []
def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1 / (1 + np.exp(-z)) # your code here


def preprocess_small():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_sample.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     - feature selection"""

    mat = loadmat('mnist_sample.mat')
    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(4996, 784))
    validation_preprocess = np.zeros(shape=(1000, 784))
    test_preprocess = np.zeros(shape=(996, 784))
    train_label_preprocess = np.zeros(shape=(4996,))
    validation_label_preprocess = np.zeros(shape=(1000,))
    test_label_preprocess = np.zeros(shape=(996,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 100  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[100:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 100] = tup[tup_perm[0:100], :]
            validation_len += 100

            validation_label_preprocess[validation_label_len:validation_label_len + 100] = label
            validation_label_len += 100

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    dimensions = train_data.shape
    feature_set = dimensions[1]
    to_be_deleted = []
    for feature_identifier in range(0, feature_set):
       unique_data, indices = np.unique(train_data[:,feature_identifier], return_index=True)
       if(unique_data.size <= 5):
          to_be_deleted.append(feature_identifier)
    train_data = np.delete(train_data, to_be_deleted, axis=1)
    validation_data = np.delete(validation_data, to_be_deleted, axis=1)
    test_data = np.delete(test_data, to_be_deleted, axis=1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            #print(tup)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.

    dimensions = train_data.shape
    feature_set = dimensions[1]
    to_be_deleted = []
    global selected_features
    for feature_identifier in range(0, feature_set):
       unique_data, indices = np.unique(train_data[:,feature_identifier], return_index=True)
       if(unique_data.size <= 3):
          to_be_deleted.append(feature_identifier)
       else:
          selected_features.append(feature_identifier)
    train_data = np.delete(train_data, to_be_deleted, axis=1)
    validation_data = np.delete(validation_data, to_be_deleted, axis=1)
    test_data = np.delete(test_data, to_be_deleted, axis=1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    training_data_size     = len(training_data)
    feature_vector_size    = len(training_data[0])
    training_data_reshaped = np.reshape(training_data, (training_data_size, feature_vector_size))
    # Adds 1 to the Input - This node acts as bias node
    training_data_with_bias = np.insert(training_data_reshaped, feature_vector_size ,1, axis=1)
    #data_size = training_data_size
    aj = np.dot(w1, training_data_with_bias.T)  # sum of weights and input
    zj = sigmoid(aj)
    biased_z_node = 1
    #Adding bias to the hidden nodes
    zj_bias = np.insert(zj, len(zj), biased_z_node, axis=0)
    bl = np.dot(w2, zj_bias)  #10 output values
    ol = sigmoid(bl)
    #epsilon_l_list = np.copy(ol)
    #Creating 1 of K encoding for True Lables 
    yl = np.zeros((training_data_size, n_class))
    yl[np.arange(training_data_size), training_label.astype(int)] = 1
    yl = yl.T
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    weight_delete_bias = np.delete(w2,len(w2[0])- 1,axis=1)
    # Gradient Descent

    #Derivative of weight2 - Output to Hidden
    #epsilon_l_list = ol - yl
    epsilon_l_list = np.subtract(ol, yl)  # ol-yl
    weight_derivative_2 = np.dot(epsilon_l_list,zj_bias.T)# del l and zj
    weight_derivative_2_lambda = np.multiply(lambdaval,w2)
    weight_derivative_2_final = np.divide(np.add(weight_derivative_2, weight_derivative_2_lambda), training_data_size)
    #weight_derivative_2_final = weight_derivative_2_intermediate / training_data_size

    #Derivative of weight1 - Hidden to Input layer
    zj_sub = np.subtract(1,zj)
    zj_derivative = np.multiply(zj_sub,zj)  # zj * (1-zj)
    summation_del_weight = np.dot(epsilon_l_list.T,weight_delete_bias)
    mul_del_weight = np.multiply(zj_derivative.T, summation_del_weight)
    weight_derivative_1 = np.dot(mul_del_weight.T, training_data_with_bias)
    weight_derivative_1_lambda = np.multiply(lambdaval, w1)
    weight_derivative_1_final = np.divide((np.add(weight_derivative_1, weight_derivative_1_lambda)), training_data_size)
    #weight_derivative_1_final = weight_derivative_1_intermediate / training_data_size
    obj_grad = np.array([])
    obj_grad = np.concatenate((weight_derivative_1_final.flatten(), weight_derivative_2_final.flatten()), 0)

    #Calculating Error function
    log_ol = np.log(ol)
    yprodol = np.multiply(yl, log_ol)
    one_minus_yl = np.subtract(1 , yl)
    one_minus_ol = np.subtract(1 , ol)
    lnoil = np.log(one_minus_ol)
    lnoil_dr = np.multiply(one_minus_yl, lnoil)
    error_summation = np.add(yprodol,lnoil_dr)
    error_summation_value = np.divide(np.sum(error_summation), training_data_size)
    #Adding a negative sign to the summation value
    obj_value_normalized = 0 - error_summation_value
    squared_w1 = np.square(w1)
    squared_w2 = np.square(w2)
    summation_w1 = np.sum(squared_w1)
    summation_w2 = np.sum(squared_w2)
    Regularazation_parameter = np.divide(lambdaval, np.multiply(2, training_data_size))
    obj_value_adjustments = np.multiply(Regularazation_parameter,(summation_w1+summation_w2))
    obj_val = obj_value_normalized + obj_value_adjustments
    #obj_val = obj_value_normalized
    #print(obj_val)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    labels = np.array([])
    # Your code here
    data_size = len(data)
    feature_vector_size = len(data[0])
    training_data_reshaped = np.reshape(data, (data_size, feature_vector_size))
    #Adds 1 to the Input - This node acts as bias node
    training_data_with_bias = np.insert(training_data_reshaped, feature_vector_size ,1, axis=1)
    aj = np.dot(w1, training_data_with_bias.T)  # calculated in hidden layer, added bias
    zj = sigmoid(aj)  # output from hidden layer after activation
    biased_z_node = 1
    #Adding bias to the hidden nodes
    zj_bias = np.insert(zj, len(zj), biased_z_node, axis=0)
    bj = np.dot(w2, zj_bias)  # calculated in output layer
    ol = sigmoid(bj)  # output from output layer after activation

    labels = np.argmax(ol, axis=0)
    return labels


"""**************Neural Network Script Starts here********************************"""
#global selected_features
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess_small()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.
#time1 = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#time2 = time.time()
# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
obj = [selected_features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj, open('params.pickle', 'wb'))
#print('Time Taken ' + str(time2 - time1))
