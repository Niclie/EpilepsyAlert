import numpy as np
from src.data_preprocessing import load_data
from src.utils import constants
from src.data_preprocessing.preprocess import make_dataset


def get_dataset(patient_id, load_from_file=True, verbose=True, split=True):
    """
    Get the dataset for a given patient.

    Args:
        patient_id (str): the ID of the patient.
        load_from_file (bool, optional): whether to load the dataset from a file. Defaults to True.

    Returns:
        dict: the dataset for the patient.
    """
    patient = load_data.load_summary_from_file(patient_id)

    if load_from_file:
        try:
            npz = np.load(f'{constants.DATASETS_FOLDER}/{patient.id}.npz')
            data = {k: npz.get(k) for k in npz}
            npz.close()
        except:
            if verbose: print(f'Dataset for {patient.id} not found')
            return None
    else:
        data = make_dataset(patient, split=split)
    
    if verbose:
        if 'train_data' in data.keys():
            print(f'Training data shape: {data['train_data'].shape}')
            print(f'Test data shape: {data['test_data'].shape}')
        else:
            print(f'Data shape: {data['data'].shape}')

    return data