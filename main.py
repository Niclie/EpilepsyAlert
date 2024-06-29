# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#ictal state     : periodo di tempo durante la crisi.  Durata: variabile
#postictal state : periodo di tempo dopo la crisi.     Durata: da definire
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore


from read_data import load_summaries_from_folder


def main():
    patients = load_summaries_from_folder()
    #print(patients[2].get_seizures_datetimes())
    print(patients[2].get_interictal_preictal_datetimes())

if __name__ == '__main__':
    main()