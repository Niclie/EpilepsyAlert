# import numpy as np
# import mne
# import pandas as pd
# #from pyedflib import highlevel
# #from sklearn.neural_network import MLPClassifier
# #import pathlib
# import time

# DATA_FOLDER = 'data'
# N_PATIENTS = 8 # Number of patients in the dataset
# N_RECORDINGS = [42, 38, 19, 19, 25, 29, 33, 31] # Number of recordings for each patient
# # TOTALE 284 FILE DI CUI:
# # file .edf = 42 + 38 + 19 + 19 + 25 + 29 + 33 + 31 = 236
# # file .seizure = 7 + 7 + 3 + 3 + 7 + 6 + 4 + 3 = 40
# # file summary = 8
# PATIENTS_ID = [1, 3, 7, 9, 10, 20, 21, 22] # ID of the patients in the dataset

# RECORDINGS_ID = [
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 30, 31, 38, 89],
#     [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 34, 59, 60, 68],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 38, 51, 54, 77]
#     ]

# def eeg_segmentation(patient, recording):
#     edf_path = f'{DATA_FOLDER}/chb{patient:02d}/chb{patient:02d}_{recording:02d}.edf'
#     raw = mne.io.read_raw_edf(edf_path, verbose='ERROR')
#     print(raw.times.shape)

# if __name__ == '__main__':
#     eeg_segmentation(1, 1)

# # edf_path = 'dataset/chb01/chb01_01.edf'
# # raw = mne.io.read_raw_edf(edf_path)
# # print(raw.times)
# # df = raw.to_data_frame()
# # df.to_csv('chb01_01.csv', index=False)
# # print(df.info())


# # def make_divisible_by_5(number):
# #     remainder = number % 1280
# #     if remainder == 0:
# #         return number
    
# #     return number - remainder
    
# # dataset_path = pathlib.Path(DATASET_FOLDER).absolute()

# # dataset = []
# # for i in range(N_PATIENTS):
# #     for j in range(N_RECORDINGS[i]):
# #         print('starting patient:', f'chb{PATIENTS_ID[i]:02d}', 'recording:', f'{RECORDINGS_ID[i][j]:02d}')
# #         signal = np.array(highlevel.read_edf(str(dataset_path / f'chb{PATIENTS_ID[i]:02d}' / f'chb{PATIENTS_ID[i]:02d}_{RECORDINGS_ID[i][j]:02d}.edf'),
# #                              verbose=True)[0])
# #         #signal = np.round(signal[:, :], 5)
# #         print('(channels, samples):',signal.shape)
        
# #         new_len = int(make_divisible_by_5(len(signal[0])))
# #         if new_len < len(signal[0]):
# #             signal = signal[:, :new_len]
# #             print('cutting signal to', new_len, 'samples')
# #         n_segments = int(new_len / 1280)
# #         print('segment each channel in', n_segments, 'segments')

# #         segmented_array = signal.reshape(23, n_segments, 1280)
# #         for k in range(n_segments):
# #             sample = np.transpose(segmented_array[:, k, :])
# #             sample = sample.ravel()
# #             dataset.append(sample)
# #         print('done\n')

# # np.save('dataset.npy', dataset)
# # np.savetxt('dataset.csv', dataset, delimiter=',')
# # NON FUNZIONA