
def _readData(filename):
  """ Returns a matrix with the data from the file. """
  f = open(filename, 'r')
  lines = f.readlines()
  # Pre-proccesses the data by removing the semicolons
  mp = lambda l : l.split(',')
  data = map(mp, lines)
  return data

def _computeDissimilarityMatrix(data):
  """ Computes the Dissimilarity Matrix. The data should be pre-proccessed. """

  # Dissimmilarity function
  delta = lambda (x_ik, x_jk) : 0 if (x_ik == x_jk) else 1
  d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j)))

  # Number of examples
  n = len(data)

  matrix = []
  for i in range(n):
    matrix.append([])
    for j in range(n):
      matrix[i].append(d(data[i], data[j]))

  return matrix

def proccessData(filename):
  data = _readData(filename)
  return _computeDissimilarityMatrix(data)
