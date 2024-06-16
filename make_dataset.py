import read_data
import os
from pathlib import Path
import mne
import pandas as pd

#The seizure prediction problem is formulated as a classification task between interictal and preictal brain states, in which a true alarm is considered when the preictal state is detected within the predetermined preictal period. There is no standard duration for the preictal state. In our experiments, the preictal duration was chosen to be one hour before the seizure onset and interictal duration was chosen to be at least four hours before or after any seizure.

DATASET_FOLDER = 'dataset'   

# def make_dataset(patient, output_folder=DATASET_FOLDER):
#     print(f'Creating dataset for patient {patient.id}')
#     recordings = patient.recordings
#     for r in recordings:
        

# def edf_to_csv(patient, recording, data_folder=read_data.DATA_FOLDER):
#     print(f'Converting {recording.id} to csv...')
#     raw = mne.io.read_raw_edf(f'{data_folder}/{patient.id}/{recording.id}.edf', include=recording.channels, verbose='ERROR')
#     df = raw.to_data_frame(index='time')
#     df['preictal'] = 0
#     min_preictal_start = 0
#     for i in range(recording.n_seizures):
#         df.loc[max(min_preictal_start, recording.seizures[i][0] - 3600 - 0.00390625):recording.seizures[i][0] - 0.00390625, 'preictal'] = 1
#         min_preictal_start = recording.seizures[i][1] + 0.00390625

#     df.to_csv(f'{DATASET_FOLDER}/{patient.id}/{recording.id}.csv')
#     print('done')
#     return True

# def make_dataset(patient, dataset_folder=DATASET_FOLDER):
#     print(f'Creating dataset for patient {patient.id}')
#     os.makedirs(f'{dataset_folder}/{patient.id}', exist_ok=True)
#     for r in patient.recordings:
#         edf_to_csv(patient, r)
#     print()


# def unify_csvs(patient, dataset_folder=DATASET_FOLDER):
#     print(f'Unifying csvs for patient {patient.id}')
#     dfs = []
#     for r in patient.recordings:
#         print(f'Unifying {r.id}...')
#         df = pd.read_csv(f'{dataset_folder}/{patient.id}/{r.id}.csv', index_col='time')
#         dfs.append(df)
#     unified_df = pd.concat(dfs)
#     unified_df.to_csv(f'{dataset_folder}/{patient.id}.csv')
#     print('done')

# def make_all_datasets(patients, dataset_folder=DATASET_FOLDER):
#     for p in patients:
#         #make_dataset(p, dataset_folder)
#         unify_csvs(p, dataset_folder)

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