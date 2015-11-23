# Author: George Oliveira (ghao@cin.ufpe.br) - 2015
#
# References:
# - https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
# - http://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node146.html

import random
from math import tanh

random.seed(42)

def sigmoid(x):
    """ Sigmoid function. """
    return 1.0 / (1.0 + exp(-x))

def mse(h, y):
    """
        Mean Squared Error.

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

def train(X, Y, testX, testY, learning_rate, hidden_units, output_units, max_iterations=200, threshold=0.05):
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
            - threshold : MSE limit.
    """
    dim = len(X[0]) # inputs dimension

    layers = 1 + len(hidden_units) + 1 # input layer + hidden layer + output layer
    units = [dim] + hidden_units + [output_units]

    W = [] # Weights matrix
    b = [] # Bias matrix

    M = [] # Momentum
    alpha = 0.001 # Momentum influence

    # Initializes W and b with random values
    for l in range( layers-1 ): # except for output
        W.append( [] )
        M.append( [] )
        b.append( [] )
        for i in range( units[l] ) :
            W[l].append([])
            M[l].append([])
            for j in range( units[l+1] ) :
                b[l].append( random.uniform(0, 0.2) ) # sets random biases
                W[l][i].append( random.uniform(0, 0.2) ) # sets a random weight W[l][i][j] from unit i in layer l to unit j in layer l+1
                M[l][i].append( 0 ) # the momentum term has no influence on the first iteration

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
                    sm = -b[l-1][j]
                    for i in range(units[l-1]):
                        sm += a[l-1][i]*W[l-1][i][j]
                    out = sm if l == layers-1 else tanh(sm) # the output layer is linear
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
                    delta[l][i] = sm * a[l][i] * (1 - a[l][i])

            for l in range(layers-1):
                for i in range(units[l]):
                    for j in range(units[l+1]):
                        # Gradient Descent
                        W[l][i][j] += learning_rate * a[l][i] * delta[l+1][j] + alpha*M[l][i][j]
                        M[l][i][j] = learning_rate * a[l][i] * delta[l+1][j]
                        b[l][j] += learning_rate * delta[l+1][j]

        error = netMSE(testX, testY, W, b, hidden_units, output_units)

        if it%100 == 0:
            print "Step #%d" % (it+1)
            print "Error %f" % (error*100)

        if error < threshold:
            break

    return (W, b)

def fit(X, W, b, hidden_units, output_units):
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
                out = sm if l == layers-1 else tanh( sm )
                a[l].append( out )
        o.append(a[layers-1])

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

def normalize(X, Y, num_classes=0):
    """ Pre-processes the data that will be used to train the network creating bit lists for the classes. """
    for i in range(len(X)):
        c = [0 for k in range(num_classes)]
        c[Y[i][0]] = 1
        Y[i] = c

    return (X, Y)

def getClass(class_bits):
    """ Returns the class that corresponds to the bit list. """
    for c in range(len(class_bits)):
        if class_bits[c] == 1:
            return c

def readData(filename):
    """ Reads a dataset from `filename` and returns it as a matrix. """
    f = open(filename, 'r')
    lines = f.readlines()
    # Pre-proccesses the data by removing the semicolons
    mp = lambda l : l.split(',')
    data = map(mp, lines)
    getX = lambda l : l[:len(l)-1]
    X = map(getX, data)
    getY = lambda l : l[-1][:-1] # Gets the last element (the class parameter)
    Y = map(getY, data)
    return (X, Y)

def proccessData(filename):
    """ Maps the input data to the model defined at the documentation and divides it into the experiments sets (train, validation and test). """
    # Dataset info:
    #
    # Total: 958 data points
    # 2 classes (positive, negative)
    #
    # - positive: 626
    # - negative: 332
    #
    # Experiments division:
    #
    # - positive:
    #     => Train: 313
    #     => Val: 157
    #     => Test: 155
    #
    # - negative:
    #     => Train: 166
    #     => Val: 83
    #     => Test: 83
    (X, Y) = readData(filename)

    mp = lambda x : 1 if x == 'x' else 0 if x == 'o' else -1
    proccessX = lambda l : map(mp, l)
    X = map(proccessX, X)

    proccessY = lambda y : [1] if y == 'positive' else [0]
    Y = map(proccessY, Y)

    (X, Y) = normalize(X, Y, num_classes=2) # processes Y to classes bit list

    C = [[], []]
    for i in range(len(Y)):
        if Y[i] == [1, 0]:
            C[0].append(i)
        else:
            C[1].append(i)

    random.shuffle(C[0])
    random.shuffle(C[1])

    negative = {'train':C[0][:166], 'validation':C[0][166:249], 'test':C[0][249:]}
    positive = {'train':C[1][:313], 'validation':C[1][313:470], 'test':C[1][470:]}

    trainData = negative['train']+positive['train']
    validationData = negative['validation']+positive['validation']
    testData = negative['test']+positive['test']

    random.shuffle(trainData)
    random.shuffle(validationData)
    random.shuffle(testData)

    dataset = {'train':([],[]), 'validation':([],[]), 'test':([],[])}

    for i in range(len(trainData)):
        pos = trainData[i]
        dataset['train'][0].append(X[pos])
        dataset['train'][1].append(Y[pos])

    for i in range(len(validationData)):
        pos = validationData[i]
        dataset['validation'][0].append(X[pos])
        dataset['validation'][1].append(Y[pos])

    for i in range(len(testData)):
        pos = testData[i]
        dataset['test'][0].append(X[pos])
        dataset['test'][1].append(Y[pos])

    return dataset

def loadData():
    """ Reads and pre-processes the dataset. """
    return proccessData('tic-tac-toe.data.txt')

if __name__ == "__main__":
    # (dataX, dataY) = loadData()
    #
    # trainX = dataX[:500]
    # trainY = dataY[:500]
    #
    # testX = dataX[501:]
    # testY = dataY[501:]
    #
    # hidden_units = [4, 15, 4]
    # learning_rate = 0.1
    #
    # (W, b) = train(trainX, trainY, testX, testY, learning_rate, hidden_units, 2, 500)

    # P = fit(trainX, W, b, hidden_units, 2, False)
    #
    # for i in range(len(P)):
    #     print "Input: "+str(testX[i])
    #     print "Predict "+str(P[i])
    #     print "Expected "+str(testY[i])
