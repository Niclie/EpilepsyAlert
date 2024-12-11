from src.preprocessing.load_data import load_summary_from_file
from src.preprocessing.dataset import Dataset, load_dataset


def run_preprocessing(patient_id, load_from_file=True, verbose=True, split=True):
    """
    Run the preprocessing pipeline for a given patient_id.

    Args:
        patient_id (str): Patient identifier.
        load_from_file (bool, optional): Whether to load the dataset from a file. Defaults to True.
        verbose (bool, optional): Whether to print the size of the dataset. Defaults to True.
        split (bool, optional): Whether to split the dataset into training and test sets. Defaults to True.

    Returns:
        dict: Dictionary with the dataset.
    """
    if load_from_file:
        data = load_dataset(patient_id)
    else:
        patient = load_summary_from_file(patient_id)
        data = Dataset(patient, split=split).make_dataset()

    if verbose:
        if 'train_data' in data.keys():
            print(f'Training data size: {len(data['train_data'])}')
            print(f'Test data size: {len(data['test_data'])}')
        else:
            print(f'Data size: {len(data['data'])}')

    return data