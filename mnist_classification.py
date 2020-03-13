from mnist import MNIST
import numpy as np


# X  -> 784 x n
#  ==> (W1): 16 x 784
#  ==> (b1): 1 x 16
# H1 -> 16
#  ==> (W2): 16 x 16
#  ==> (b2): 1 x 16
# H2 -> 16
#  ==> (W3): 10 x 16
#  ==> (b3): 1 x 10
# y' -> 10

class Model(object):
    activations = {
        'tanh': np.tan,
        'sigmoid': lambda ndarr: 1 / (1 + np.exp(-ndarr))
    }
    
    def __init__(self, inputs, outputs, activation='tanh'):
        self.inputs = inputs
        self.outputs = outputs
        self.W1 = np.random.rand(16, 784)
        self.W2 = np.random.rand(16, 16)
        self.W3 = np.random.rand(10, 16)
        self.b1 = np.random.rand((1, 16))
        self.b2 = np.random.rand((1, 16))
        self.b3 = np.random.rand((1, 10))
        self.act_func = np.tanh()
        self.learning_rate = 0.1
        
        
    # Cost function, calculates squared distance
    # of the prediction from the labeled data
    def calculate_cost(self):
        y_hat = self.forward(self.inputs)
        return np.power(self.outputs - y_hat, 2)
        
        
    
    # The forward propagation algorithm
    # For each layer in the NN:
    # Computes Z = WX + b
    # Computes a = activation(Z)
    # And returns the final activation layer
    def forward(self, x):
        # Layer 0->1
        z1 = self.W1.dot(x) + self.b1
        a1 = self.act_func(z1)
        # Layer 1->2
        z2 = self.W2.dot(a1) + self.b2
        a2 = self.act_func(z2)
        # Layer 2->3
        z3 = self.W3.dot(a2) + self.b3
        softmax = z3 / np.sum(z3, axis=0, keepdims=True)
        return softmax
    
    # The backward propagation algorithm
    # For each epoch
    def backward(self, x, y, epochs=20000):
        pass
    
    # Makes a prediction using the forward algorithm
    # Uses the current set of weights and biases to
    # predict the class of the given input 'x'
    def predict(self, x):
        return self.forward(x)


if __name__ == '__main__':
    mndata = MNIST('samples')
    
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()
    
    model = Model(X_train, y_train)
