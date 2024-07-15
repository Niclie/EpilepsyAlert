import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.')) # to import src package
import src.data_preprocessing.load_data as load_data
import src.utils.constants as constants

# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore prima dello stato preictale


# In order to overcome the problem of the imbalanced dataset, we selected the number of interictal segments to be equal to the available number of preictal segments during the training process. The interictal segments were selected at random from the overall interictal samples.


def create_dataset(patient_ids=None, all_patients = False):
    if all_patients:
        patients = load_data.load_summaries_from_folder()
        for p in patients:
            p.make_dataset()
    else:
        for id in patient_ids:
            patient = load_data.load_summary_from_file(id)
            patient.make_dataset()


def create_dataset_all_patients():
    create_dataset(all_patients=True)
    file_names = os.listdir(constants.DATASET_FOLDER)
    for fn in file_names:
        print(f'{fn}:')
        dataset = np.load(f'{constants.DATASET_FOLDER}/{fn}')
        for k in dataset.keys():
            print(dataset.get(k).shape)
        print('\n')


def create_dataset_one_patient(id):
    create_dataset([id])
    dataset = np.load(f'{constants.DATASET_FOLDER}/{id}.npz')
    for k in dataset.keys():
        print(dataset.get(k).shape)

def main():
    #create_dataset_one_patient('chb07')
    create_dataset_all_patients()


if __name__ == '__main__':
    main()