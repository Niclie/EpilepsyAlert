import sys
import os
import pandas as pd

# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore prima dello stato preictale

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.data_preprocessing.load_data as load_data
import src.utils.constants as constants


def main():
    patients = load_data.load_summaries_from_folder()
    [patient.make_dataset() for patient in patients]

    for patient in patients:
        print(patient.id)
        try:
            df = pd.read_parquet(f'{constants.DATASET_FOLDER}/{patient.id}.parquet')
            print(df.info())
        except FileNotFoundError:
            print('No dataset found')
        print("\n\n")

if __name__ == '__main__':
    main()