# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#ictal state     : periodo di tempo durante la crisi.  Durata: variabile
#postictal state : periodo di tempo dopo la crisi.     Durata: da definire
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore


import read_data
import numpy as np

def main():
    patient = read_data.load_summary_from_file('chb01')
    print(patient.make_dataset().shape)

    # patients = read_data.load_summaries_from_folder()
    # for p in patients:
    #     for s in p.get_continuous_recording_indexes():
    #         print(p.recordings[s[0]], p.recordings[s[1]])
    #     print("\n")        

if __name__ == '__main__':
    main()