import numpy as np
from pyedflib import highlevel
from sklearn.neural_network import MLPClassifier
import pathlib

DATASET_FOLDER = 'dataset'
dataset_path = pathlib.Path(DATASET_FOLDER).absolute()



signal = np.array(highlevel.read_edf(str(dataset_path / 'chb01' / 'chb01_01.edf'))[0])

# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, y)
# print(clf.predict([[2., 2.], [-1., -2.]]))