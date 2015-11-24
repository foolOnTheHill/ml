# Author: George Oliveira (ghao@cin.ufpe.br) - 2015
#
# References:
# - https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
# - http://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node146.html

from numbers import Number

import random
from math import tanh, isnan, isinf

def mse(h, y):
    """
        Mean Squared Error.da

        Parameters:
            - h : the output given by the network;
            - y : the expected output.
    """
    d = sum([(h[i] - y[i])**2 for i in range(len(y))])
    return 0.5*d

def netMSE(X, Y, W, b, hidden_units, output_units):
    """
        Computes the MSE for the current network to perform the validation test run while training.

        Parameters:
            - X : validation data inputs;
            - Y : validation data outputs;
            - W : weights matrix of the network;
            - b : biases matrix of the network;
            - hidden_units : a list containing the number of units per hidden layer;
            - output_units : number of output units (must be equal to the dimension of the elements on Y).
    """
    dim = len(X[0]) # inputs dimension

    layers = 1 + len(hidden_units) + 1 # input layer + hidden layer + output layer
    units = [dim] + hidden_units + [output_units]

    m = len(X)
    P = fit(X, W, b, hidden_units, output_units)

    sm = sum([mse(Y[i], P[i]) for i in range(m)]) / float(m)

    return sm

def train(X, Y, testX, testY, learning_rate, hidden_units, output_units, max_iterations=200, max_fail=5, threshold=0.05, debug=False):
    """
        Uses Backpropagation Gradient Descent to train a Multi-layer Perceptron.

        Parameters:
            - X : train data inputs;
            - Y : train data outputs;
            - testX : validation inputs;
            - testY : validation outputs;
            - learning_rate : MLP learning rate;
            - hidden_units : a list containing the number of units per hidden layer;
            - output_units : number of output units (must be equal to the dimension of the elements on Y);
            - max_iterations : maximum number of iterations of the training algorithm;
            - max_fail : maximum number of consecutive increaces on the MSE;
            - threshold : MSE limit.
    """
    dim = len(X[0]) # inputs dimension

    layers = 1 + len(hidden_units) + 1 # input layer + hidden layer + output layer
    units = [dim] + hidden_units + [output_units]

    W = [] # Weights matrix
    b = [] # Bias matrix

    M = [] # Momentum
    alpha = 0.505 # Momentum influence

    # Initializes W and b with random values
    for l in range( layers-1 ): # except for output
        W.append( [] )
        M.append( [] )
        b.append( [] )
        for i in range( units[l] ) :
            W[l].append([])
            M[l].append([])
            for j in range( units[l+1] ) :
                b[l].append( random.uniform(0, 0.01) ) # sets random biases
                W[l][i].append( random.uniform(0, 0.01) ) # sets a random weight W[l][i][j] from unit i in layer l to unit j in layer l+1
                M[l][i].append( 0 ) # the momentum term has no influence on the first iteration

    # Stop creteria
    num_fails = 0
    trainError = None

    previousValidationError = None
    validationError = None

    for it in range(max_iterations):
        # print W
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
                    sm = -b[l-1][j]
                    for i in range(units[l-1]):
                        sm += a[l-1][i]*W[l-1][i][j]
                    out = tanh(sm)
                    a[l].append( out )
                    z[l].append( sm )

            # Backward step
            delta = [[0 for j in range(units[i])] for i in range(layers)]

            for i in range(output_units): # Computes delta only for the output layer
                delta[layers-1][i] = - ( opt[i] - a[layers-1][i] ) * (1 - a[layers-1][i]*a[layers-1][i])

            for l in range(layers-2, -1, -1): # Backpropagates the error
                for i in range( units[l] ):
                    sm = 0
                    for j in range( units[l+1] ) :
                        sm += W[l][i][j]*delta[l+1][j]
                    delta[l][i] = sm * (1 - a[l][i]*a[l][i])

            for l in range(layers-1):
                for i in range(units[l]):
                    for j in range(units[l+1]):
                        # Gradient Descent
                        W[l][i][j] += learning_rate * a[l][i] * delta[l+1][j] + alpha*M[l][i][j]
                        M[l][i][j] = learning_rate * a[l][i] * delta[l+1][j]
                        b[l][j] += learning_rate * delta[l+1][j]

        validationError = netMSE(testX, testY, W, b, hidden_units, output_units)
        trainError = netMSE(X, Y, W, b, hidden_units, output_units)

        if it%100 == 0 and debug:
            print "Step #%d" % (it+1)
            print "Error %f" % validationError

        # Stop criteria
        if previousValidationError == None:
            previousValidationError = validationError
        elif validationError > previousValidationError:
            num_fails += 1
            previousValidationError = validationError
        else :
            previousValidationError = None
            num_fails = 0

        if (trainError < threshold) or (num_fails == max_fail):
            if debug:
                print "Step #%d" % (it+1)
                print "Error %f" % validationError
            break

    return (W, b, trainError, validationError)

def fit(X, W, b, hidden_units, output_units, normalize_output=False):
    """
        Given the Weights matrix and the bias representing a Multi-layer Perceptron, classifies the data present on X.

        Parameters:
            - X : the data that will be classified;
            - W : the weights matrix of the network;
            - b : the biases matrix of the network;
            - hidden_units: a list containing the number of units per hidden layer;
            - output_units : number of output units on the Multi-layer Perceptron.
    """
    dim = len(X[0])
    layers = 1 + len(hidden_units) + 1
    units = [dim] + hidden_units + [output_units]

    o = []
    for k in range(len(X)):
        ipt = X[k]
        a = [ipt]
        for l in range(1, layers):
            a.append([])
            for j in range(units[l]):
                sm = -b[l-1][j]
                for i in range(units[l-1]):
                    sm += a[l-1][i]*W[l-1][i][j]
                out = tanh( sm )
                a[l].append( out )
        o.append(a[layers-1])

    if normalize_output == False:
        return o

    pred = []
    for i in range(len(X)):
        c = 0
        p = o[i][0]
        for j in range(1, output_units):
            if o[i][j] > p:
                c = j
                p = o[i][j]
        class_bits = [0 for k in range(output_units)]
        class_bits[c] = 1
        pred.append(class_bits)

    return pred
