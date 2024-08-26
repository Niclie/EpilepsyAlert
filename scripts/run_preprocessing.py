import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
import src.data_preprocessing.load_data as load_data
import src.utils.constants as constants
from src.data_preprocessing.preprocess import make_dataset, make_dataset_v2


def get_dataset(patient_id, load_from_file = True, gamma_band = False):
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
            print(f'Dataset for {patient.id} not found')
        
        return data

    return make_dataset_v2(patient, use_gamma_band=gamma_band)


def main():
    # patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19','chb20', 'chb21', 'chb22', 'chb23']

    #patients_v2 = ['chb02', 'chb03', 'chb04', 'chb11', 'chb12', 'chb13', 'chb14', 'chb17', 'chb19','chb20', 'chb23'] #, 'chb24'

    return

if __name__ == '__main__':
    main()