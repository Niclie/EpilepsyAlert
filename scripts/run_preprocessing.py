import sys
import os
sys.path.append(os.path.abspath('.')) # to import src package
import numpy as np
import src.data_preprocessing.load_data as load_data
import src.utils.constants as constants


def get_dataset(patient_ids = None, load_from_file = False):
    """
    Get the dataset of the specified patients.

    Args:
        patient_ids (list, optional): list of patient IDs if None, all the patients will be used. Defaults to None.
        load_from_file (bool, optional): if True, the data will be loaded from the files. Defaults to False.

    Returns:
        list: list of datasets.
    """
    if not patient_ids:
        patients = load_data.load_summaries_from_folder()
        if load_from_file:
            data = [np.load(f'{constants.DATASET_FOLDER}/{p.id}.npz') for p in patients]
        else:
            data = [p.make_dataset() for p in patients]
    else:
        if load_from_file:
            data = [np.load(f'{constants.DATASET_FOLDER}/{p}.npz') for p in patient_ids]
        else:
            patients = [load_data.load_summary_from_file(p) for p in patient_ids]
            data = [p.make_dataset() for p in patients]

    return data

def main():
    data = get_dataset(['chb01'])
    print(data[0].keys())

    return

if __name__ == '__main__':
    main()