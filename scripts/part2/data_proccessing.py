import random

random.seed(58)

def readData(filename):
    """ Returns a matrix with the data from the file. """
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

def getClasses(Y):
    """ Divides the data set according to elements classes. """
    C = [[], []]
    for i in range(len(Y)):
        if Y[i] == 0:
            C[0].append(i)
        else:
            C[1].append(i)
    return C

def divideData(X, Y):
    """ Divides the data into 10 holdouts. """
    H = []
    n = len(X)
    data = zip(X, Y)
    random.shuffle(data)
    X = data[0]
    Y = data[1]
    size = n / 10

    holdFile = open('holdouts.txt', 'w')

    mp = lambda x : str(x)
    jn = lambda l : ' '.join(map(mp, l))

    s = 0 # Starting point
    for i in range(10):
        e = s + size
        ipt = X[s:e]
        out = Y[s:e]
        H.append( (ipt, out) )

        holdFile.write("\nHoldout #%d\n" % (i+1))
        for j in range(len(ipt)):
            holdFile.write( jn(ipt[j]+out[j]) )
            holdFile.write('\n')

        s = e

    holdFile.close()

    return H

def proccessData(filename):
    """ Maps the input data to the model defined at the documentation. """
    (X, Y) = readData(filename)
    mp = lambda x : 1 if x == 'x' else 0 if x == 'o' else -1
    proccessX = lambda l : map(mp, l)
    X = map(proccessX, X)
    proccessY = lambda y : 1 if y == 'positive' else 0
    Y = map(proccessY, Y)
    H = divideData(X, Y)
    return H

def loadData():
    return proccessData('tic-tac-toe.data.txt')
