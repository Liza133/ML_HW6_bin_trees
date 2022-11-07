import numpy as np
from tree import DT, Tree
from dataset import CandyDataset
from node import Node

cd = CandyDataset('candy-data.csv')
X = cd.get_classification_data()[0]
target = cd.get_classification_data()[1]

# Normalisation
for i in range(X.shape[0]):
    X[i] = 2 * (X[i] - np.amin(X[i])) / (np.amax(X[i]) - np.amin(X[i])) - 1

# Shuffle data
tr = 0.7
val = 0.3
N = X.shape[0]
ind_prm = np.random.permutation(np.arange(N))
train_ind = ind_prm[:int(tr * N)]
valid_ind = ind_prm[int(tr * N):]
X_train, target_train, X_valid, target_valid = X[train_ind], target[train_ind], X[valid_ind], target[train_ind]

tree = Tree()

depth = 1
root = Node()
dt = DT(10, 0.01, 10)
dt.buildTree(X, target, root, depth, tree)
print(tree.get_tree())