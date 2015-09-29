
def readData(filename):
    """ Returns a matrix with the data from the file. """
    f = open(filename, 'r')
    lines = f.readlines()
    # Pre-proccesses the data by removing the semicolons
    mp = lambda l : l.split(',')
    data = map(mp, lines)
    getX = lambda l : l[:len(l)-1]
    X = map(getX, data)
    getY = lambda l : l[-1] # Gets the last element (the class parameter)
    Y = map(getY, data)
    return (X, Y)

def proccessData(filename):
    """ Maps the input data to the model defined at the documentation. """
    (X, Y) = readData(filename)
    mp = lambda x : 1 if x == 'x' else 0 if x == 'o' else -1
    proccessX = lambda l : map(mp, l)
    X = map(proccessX, X)
    proccessY = lambda y : 1 if y == 'positive' else 2
    Y = map(proccessY, Y)
    return (X, Y)
