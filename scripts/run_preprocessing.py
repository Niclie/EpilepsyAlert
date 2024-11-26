import numpy as np
from src.data_preprocessing.load_data import load_summary_from_file
from src.data_preprocessing.dataset import Dataset
from src.utils.constants import DATASETS_FOLDER


def get_dataset(patient_id, load_from_file=True, verbose=True, split=True):
    """
    Get the dataset for a given patient.

    Args:
        patient_id (str): the ID of the patient.
        load_from_file (bool, optional): whether to load the dataset from a file. Defaults to True.
        verbose (bool, optional): whether to print information about the dataset. Defaults to True.
        split (bool, optional): whether to split the dataset into training and test sets. Defaults to True

    Returns:
        dict: the dataset for the patient.
    """
    if load_from_file:
        try:
            npz = np.load(f'{DATASETS_FOLDER}/{patient_id}.npz')
            data = {k: npz.get(k) for k in npz}
            npz.close()
        except:
            if verbose: print(f'Dataset for {patient_id} not found')
            return None
    else:
        patient = load_summary_from_file(patient_id)
        data = Dataset(patient, split=split).make_dataset()
    
    if verbose:
        if 'train_data' in data.keys():
            print(f'Training data shape: {data['train_data'].shape}')
            print(f'Test data shape: {data['test_data'].shape}')
        else:
            print(f'Data shape: {data['data'].shape}')

    return data