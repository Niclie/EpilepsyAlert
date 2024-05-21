import numpy as np
from pyedflib import highlevel
from sklearn.neural_network import MLPClassifier
import pathlib

DATASET_FOLDER = 'dataset' # Folder containing the dataset
N_PATIENTS = 8 # Number of patients in the dataset
N_RECORDINGS = [42, 38, 19, 19, 25, 29, 33, 31] # Number of recordings for each patient
# TOTALE 284 FILE DI CUI:
# file .edf = 42 + 38 + 19 + 19 + 25 + 29 + 33 + 31 = 236
# file .seizure = 7 + 7 + 3 + 3 + 7 + 6 + 4 + 3 = 40
# file summary = 8
PATIENTS_ID = [1, 3, 7, 9, 10, 20, 21, 22] # ID of the patients in the dataset

RECORDINGS_ID = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 30, 31, 38, 89],
    [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 34, 59, 60, 68],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 38, 51, 54, 77]
    ]

dataset_path = pathlib.Path(DATASET_FOLDER).absolute()

for i in range(N_PATIENTS):
    for j in range(N_RECORDINGS[i]):
        print(highlevel.read_edf(str(dataset_path / f'chb0{i+1}' / f'chb0{i+1}_{j+1:02d}.edf'))[0].shape)

#signal = np.array(highlevel.read_edf(str(dataset_path / 'chb01' / 'chb01_01.edf'))[0])




# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, y)
# print(clf.predict([[2., 2.], [-1., -2.]]))