import random

from mlp import train, fit
from mlp_data_proccessing import loadData, getClass

def run():
    # Network configuration
    num_outputs = 2
    learning_rate = [0.3, 0.5, 0.7]
    hidden_neurons = [30, 50, 60]
    epochs = [150, 500, 2000]

    # Tic-tac-toe dataset
    dataset = loadData()

    trainingResults = open('training-results.txt', 'w')

    networks = []
    errors = []
    index = 0
    for i in range(len(learning_rate)):
        for j in range(len(epochs)):
            for k in range(len(hidden_neurons)):
                print index
                hidden_units = [hidden_neurons[k]]
                (W, b, trainError, validationError) = train(dataset['train'][0], dataset['train'][1], dataset['validation'][0], dataset['validation'][1], learning_rate[i], hidden_units, num_outputs, epochs[j])
                networks.append( (learning_rate[i], epochs[j], hidden_units, num_outputs) )
                errors.append( (validationError, trainError, index) )
                trainingResults.write("%d %f %d %d %f %f\n" % (index, learning_rate[i], epochs[j], hidden_neurons[k], validationError, trainError))
                index += 1

    trainingResults.close()

    bestNetworkResults = open('best-net-results.txt', 'w')

    best_network = networks[min(errors)[2]] # index of the network with the lowest validation error

    # Best parameters
    best_lr = best_network[0]
    best_ep = best_network[1]
    best_hd = best_network[2]
    best_no = best_network[3]

    bestNetworkResults.write('Learning rate: %f\n' % best_lr)
    bestNetworkResults.write('Epochs: %d\n' % best_ep)
    bestNetworkResults.write('Hidden units: %d\n\n' % best_hd[0])

    test_set_size = len(dataset['test'][1])

    test_networks = []
    test_errors = []
    for i in range(30):
        confusionMatrix = [[0, 0], [0, 0]]

        (W, b, trainError, validationError) = train(dataset['train'][0], dataset['train'][1], dataset['validation'][0], dataset['validation'][1], best_lr, best_hd, best_no, best_ep)
        T = fit(dataset['test'][0], W, b, best_hd, best_no, True)
        e = 0
        for k in range(test_set_size):
            if dataset['test'][1][k] != T[k]:
                e += 1
            confusionMatrix[getClass(dataset['test'][1][k])][getClass(T[k])] += 1
        e = float(e) / test_set_size

        bestNetworkResults.write("%d %f %f %f\n" % ((i+1), trainError, validationError, e))

        confFile = open('confusion-matrix-'+str(i+1)+'.txt', 'w')
        for p in range(2):
            s = ''
            for q in range(2):
                s += str(confusionMatrix[p][q]) + ' '
            s += '\n'
            confFile.write(s)
        confFile.close()

        test_networks.append( (W, b, best_hd, best_no) )
        test_errors.append( (e, trainError, validationError, i) )

    bestNetworkResults.close()

    best_test_network = min(test_errors)[3]

if __name__ == "__main__":
    run()
    # dataset = loadData()
    #
    # learning_rate = 0.3
    # hidden_units = [20, 30, 20]
    #
    # (W, b, trainError, validationError) = train(dataset['train'][0], dataset['train'][1], dataset['validation'][0], dataset['validation'][1], learning_rate, hidden_units, 2, 500, debug=True)
    #
    # P = fit(dataset['test'][0], W, b, hidden_units, 2, True)
    #
    # e = 0
    # for i in range(len(P)):
    #     if P[i] != dataset['test'][1][i]:
    #         e += 1
    # error = float(e) / len(P)
    # print "Test error %f" % error
