import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
import src.data_preprocessing.load_data as load_data
import src.utils.constants as constants


def get_dataset(patient_id, load_from_file = False, split_dataset = True):
    """
    Get the dataset for a given patient.

    Args:
        patient_id (str): the ID of the patient.
        load_from_file (bool, optional): whether to load the dataset from a file. Defaults to False.
        split_dataset (bool, optional): whether to split the dataset. Defaults to True.

    Returns:
        dict: the dataset for the patient.
    """

    patient = load_data.load_summary_from_file(patient_id)

    if load_from_file:
        data = {}
        try:
            npz = np.load(f'{constants.DATASETS_FOLDER}/{patient.id}.npz')
            data = {k: npz.get(k) for k in npz}
            npz.close()
        except:
            print(f'Dataset for {patient.id} not found')
        print(f'Dataset shape: {data[list(data.keys())[0]].shape}')
        return data
    elif split_dataset:
        return patient.make_dataset()

    return patient.make_dataset(split = False)