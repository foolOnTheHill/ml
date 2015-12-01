import random

from mlp import train, fit
from mlp_data_proccessing import loadData

def run():
    # Network configuration
    num_outputs = 2
    learning_rate = []
    epochs = []
    hidden_neurons = []

    # Tic-tac-toe dataset
    dataset = loadData()

    networks = []
    errors = []
    index = 0
    for i in range(len(learning_rate)):
        for j in range(len(epochs)):
            for k in range(len(hidden_neurons)):
                hidden_units = [20, hidden_neurons[k], 20]
                (W, b, trainError, validationError) = train(dataset['train'][0], dataset['train'][1], dataset['validation'][0], dataset['validation'][1], learning_rate[i], hidden_units, num_outputs, epochs[j])
                networks.append( (learning_rate[i], epochs[j], hidden_units, num_outputs) )
                errors.append( (validationError, trainError, index) )
                index += 1

    best_network = networks[min(errors)[2]] # index of the network with the lowest validation error

    # Best parameters
    best_lr = networks[best_network][0]
    best_ep = networks[best_network][1]
    best_hd = networks[best_network][2]
    best_no = networks[best_network][3]

    test_set_size = len(dataset['test'][1])

    test_networks = []
    test_errors = []
    for i in range(30):
        (W, b, trainError, validationError) = train(dataset['train'][0], dataset['train'][1], dataset['validation'][0], dataset['validation'][1], best_lr, best_hd, best_no, best_ep)
        T = fit(dataset['test'][0], W, b, best_hd, best_no, True)
        e = 0
        for k in range(test_set_size):
            if dataset['test'][1][k] != T[k]:
                e += 1
        e = float(e) / test_set_size

        test_networks.append( (W, b, best_hd, best_no) )
        test_errors.append( (e, trainError, validationError, i) )

    best_test_network = min(test_errors)[3]

if __name__ == "__main__":
    dataset = loadData()

    learning_rate = 0.3
    hidden_units = [20, 30, 20]

    (W, b, trainError, validationError) = train(dataset['train'][0], dataset['train'][1], dataset['validation'][0], dataset['validation'][1], learning_rate, hidden_units, 2, 500, debug=True)

    P = fit(dataset['test'][0], W, b, hidden_units, 2, True)

    e = 0
    for i in range(len(P)):
        if P[i] != dataset['test'][1][i]:
            e += 1
    error = float(e) / len(P)
    print "Test error %f" % error