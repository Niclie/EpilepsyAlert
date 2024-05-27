import read_data
import os
from pathlib import Path
import mne
#import pandas as pd

#The seizure prediction problem is formulated as a classification task between interictal and preictal brain states, in which a true alarm is considered when the preictal state is detected within the predetermined preictal period. There is no standard duration for the preictal state. In our experiments, the preictal duration was chosen to be one hour before the seizure onset and interictal duration was chosen to be at least four hours before or after any seizure.

DATASET_FOLDER = 'dataset'

# def make_dataset(patient):
#     os.makedirs(f'{DATASET_FOLDER}/{patient.id}', exist_ok=True)
#     for r in patient.recordings:
#         raw = mne.io.read_raw_edf(f'{read_data.DATA_FOLDER}/{patient.id}/{r.id}.edf', verbose='ERROR')
#         df = raw.to_data_frame(index='time')
#         df['preictal'] = 0
#         for i in range(r.n_seizures):
#             df.loc[max(0, r.seizures[i][0] - 3600):r.seizures[i][0], 'preictal'] = 1

#         df.to_csv(f'{DATASET_FOLDER}/{patient.id}/{r.id}.csv')    

def edf_to_csv(patient, recording, data_folder=read_data.DATA_FOLDER):
    print(f'Converting {recording.id} to csv...')
    raw = mne.io.read_raw_edf(f'{data_folder}/{patient.id}/{recording.id}.edf', verbose='ERROR')
    df = raw.to_data_frame(index='time')
    df['preictal'] = 0
    for i in range(recording.n_seizures):
        df.loc[max(0, recording.seizures[i][0] - 3600):recording.seizures[i][0], 'preictal'] = 1

    df.to_csv(f'{DATASET_FOLDER}/{patient.id}/{recording.id}.csv')
    print('done')
    return True

def make_dataset(patient, dataset_folder=DATASET_FOLDER):
    print(f'Creating dataset for patient {patient.id}')
    os.makedirs(f'{dataset_folder}/{patient.id}', exist_ok=True)
    for r in patient.recordings:
        edf_to_csv(patient, r)
    print()

# def main():
    # patients = read_data.read_data()
    # for p in patients:
    #     recordings = p.recordings
    #     for r in recordings:
    #         edf_to_csv(p, r)
            # path = f'{read_data.DATA_FOLDER}/{p.id}/{r.id}.edf'
            # raw = mne.io.read_raw_edf(path, verbose='ERROR')
            # df = raw.to_data_frame(index='time')
            # df['preictal'] = 0
            # for i in range(r.n_seizures):
            #     df.loc[max(0, r.seizures[i][0] - 3600):r.seizures[i][0], 'preictal'] = 1

            # os.makedirs(f'{DATASET_FOLDER}/{p.id}', exist_ok=True)
            # output_path = f'{DATASET_FOLDER}/{p.id}/{r.id}.csv'
            # df.to_csv(output_path)
            # exit(1)