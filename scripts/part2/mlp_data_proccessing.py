import random

random.seed(58)

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

    trainFile = open('train.txt', 'w')
    validationFile = open('validation.txt', 'w')
    testFile = open('test.txt', 'w')

    mp = lambda x : str(x)
    jn = lambda l : ' '.join(map(mp, l))

    for i in range(len(trainData)):
        pos = trainData[i]
        dataset['train'][0].append(X[pos])
        dataset['train'][1].append(Y[pos])
        trainFile.write( jn(X[pos]+Y[pos]) )
        trainFile.write('\n')

    for i in range(len(validationData)):
        pos = validationData[i]
        dataset['validation'][0].append(X[pos])
        dataset['validation'][1].append(Y[pos])
        validationFile.write( jn(X[pos]+Y[pos]) )
        validationFile.write('\n')

    for i in range(len(testData)):
        pos = testData[i]
        dataset['test'][0].append(X[pos])
        dataset['test'][1].append(Y[pos])
        testFile.write( jn(X[pos]+Y[pos]) )
        testFile.write('\n')

    trainFile.close()
    validationFile.close()
    testFile.close()

    return dataset

def loadData():
    """ Reads and pre-processes the dataset. """
    return proccessData('tic-tac-toe.data.txt')
