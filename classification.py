import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import math


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
    
# Data set we use:
# X - (n, 2)
# y - (n, )
X, y = sklearn.datasets.make_moons(200, noise=0.2)


# Currently used activation function
activation_sigmoid = lambda x: 1/(1+ math.exp(-1 * x))
activation_relu = lambda x: max(0, x)
activation_tanh = math.tanh

# CURRENTLY USED ACTIVATION
current_activation = activation_tanh


# Parameters of our NN:
num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Maps the given function to every element of the array
def map_to_array(f, arr):
    return np.vectorize(f)(arr)

# Builds the model with the given number of layers
def build_model(nn_hdim, num_passes=20000, print_loss=True):
    
    np.random.seed(0)
    
    # Randomly initialize our weights and biases.
    # Currently, all weights are random and all biases are 0
    W1 = np.random.rand(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.rand(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    model = {}
    
    # Gradient descent for each batch
    for i in range(0, num_passes):
        
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = map_to_array(current_activation, z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    
        # Back propagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # Add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
    
    
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
    
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
    if print_loss and (i % 1000 == 0):
        print(f"Loss after iteration {i}: {calculate_loss(model)}")
        
    return model


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = map_to_array(current_activation, z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
    
    
    
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation to calculate predictions
    z1 = X.dot(W1) + b1
    a1 = map_to_array(current_activation, z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculate loss
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


if __name__ == '__main__':
    model = build_model(5)
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Decision Boundary for hidden layer size 5")
    plt.show()
    
