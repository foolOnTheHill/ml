# Author: George Oliveira (ghao@cin.ufpe.br) - 2015
#
# Vaguely based on: https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
#
# TODO:
# - Create a MLP class;
# - Create accuracy measures;

import random
# from sklearn import datasets, metrics
from math import exp

random.seed(42)

def sigmoid(x):
    """ Sigmoid function. """
    return 1.0 / (1.0 + exp(-x))

def deriv(x):
    """ Derivative of the Sigmoid function. """
    return sigmoid(x)*(1 - sigmoid(x))

def train(X, Y, learning_rate, hidden_units, output_units, max_iterations):
    """
        Uses Backpropagation Gradient Descent to train a Multi-layer Perceptron.

        Parameters:
            - X : Train data inputs;
            - Y : Train data outputs;
            - learning_rate : MLP learning rate;
            - hidden_units : a list containing the number of units per hidden layer;
            - output_units : number of output units (must be equal to the dimension of the elements on Y);
            - max_iterations : maximum number of iterations of the training algorithm.
    """
    dim = len(X[0]) # inputs dimension

    layers = 1 + len(hidden_units) + 1 # input layer + hidden layer + output layer
    units = [dim] + hidden_units + [output_units]

    W = [] # Weights matrix
    b = [] # Bias matrix

    # Initializes W and b with random values
    for l in range( layers-1 ): # except for output
        W.append([])
        b.append( random.uniform(0, 1) )
        for i in range( units[l] ) :
            W[l].append([])
            for j in range( units[l+1] ) :
                W[l][i].append( random.uniform(0, 1) ) # sets a random weight W[l][i][j] from unit i in layer l to unit j in layer l+1

    b.append( random.uniform(0, 1) ) # bias for output layer

    for it in range(max_iterations):
        for k in range(len(X)):
            ipt = X[k]
            opt = Y[k]

            # Forward step
            a = [ipt]
            z = [ipt]
            for l in range(1, layers):
                a.append([])
                z.append([])
                for j in range(units[l]):
                    sm = -b[l]
                    for i in range(units[l-1]):
                        sm += a[l-1][i]*W[l-1][i][j]
                    a[l].append( sigmoid(sm) )
                    z[l].append( sm )

            # Backward step
            delta = [[0 for j in range(units[i])] for i in range(layers)]

            for i in range(output_units): # Computes delta only for the output layer
                delta[layers-1][i] = - ( opt[i] - a[layers-1][i] ) * deriv( z[layers-1][i] )

            for l in range(layers-2, 0, -1): # Backpropagates the error
                for i in range( units[l] ):
                    sm = 0
                    for j in range( units[l+1] ) :
                        sm += W[l][i][j]*delta[l+1][j]
                    delta[l][i] = sm * deriv( z[l][i] )

            for l in range(layers-1):
                for i in range(units[l]):
                    for j in range(units[l+1]):
                        # Gradient Descent
                        W[l][i][j] -= learning_rate * a[l][i] * delta[l+1][j]
                        b[l] -= learning_rate * delta[l+1][j]

    return (W, b)

def classify(X, W, b, hidden_units, output_units):
    """
        Given the Weights matrix and the bias representing a Multi-layer Perceptron, classifies the data present on X.

        Parameters:
            - X : the data that will be classified;
            - W : the weights matrix;
            - b : the bias for each layer;
            - hidden_units: a list containing the number of units per hidden layer;
            - output_units : number of output units on the Multi-layer Perceptron.
    """
    dim = len(X[0])
    layers = 1 + len(hidden_units) + 1
    units = [dim] + hidden_units + [output_units]

    for k in range(len(X)):
        ipt = X[k]
        a = [ipt]
        for l in range(1, layers):
            a.append([])
            for j in range(units[l]):
                sm = -b[l]
                for i in range(units[l-1]):
                    sm += a[l-1][i]*W[l-1][i][j]
                a[l].append( sigmoid(sm) )

    return a[layers-1]

# # Uncomment if you have 'sklearn'
# if __name__ == "__main__":
#     iris = datasets.load_iris()
#
#     to_list = lambda x : [x]
#
#     iris.target = map(to_list, iris.target)
#
#     trainX = iris.data[:135]
#     trainY = iris.target[:135]
#
#     testX = iris.data[136:]
#     testY = iris.target[136:]
#
#     (W, b) = train(trainX, trainY, 0.1, [10, 20, 10], 1, 200)
#
#     pred = classify(testX, W, b, [10, 20, 10], 1)
#
#     error = 0
#     for i in range(len(pred)):
#         if pred[i] != testY[i]:
#             error += 1
#
#     score =  error / float(len(testX))
#     print "Accuracy %f" % score
