from src.data_preprocessing.load_data import load_summary_from_file

a = load_summary_from_file('chb01')
for i in a.recordings:
    print(i, "\n")