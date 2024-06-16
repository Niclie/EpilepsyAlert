#n questo lavoro viene proposta una nuova tecnica di previsione delle crisi epilettiche !!!specifica per il paziente!!!, basata sull'apprendimento profondo e applicata alle registrazioni a lungo termine dell'elettroencefalogramma (EEG) del cuoio capelluto.

#In questo studio abbiamo scelto otto soggetti in modo che i periodi interictali e preictali prestabiliti fossero soddisfatti, le registrazioni non fossero interrotte e fossero disponibili le registrazioni di tutti i canali.

#Per superare il problema dello squilibrio del set di dati, durante il processo di addestramento abbiamo selezionato un numero di segmenti interictali pari al numero di segmenti preictali disponibili. I segmenti interictali sono stati selezionati a caso dai campioni interictali complessivi.

# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#ictal state     : periodo di tempo durante la crisi.  Durata: variabile
#postictal state : periodo di tempo dopo la crisi.     Durata: da definire
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore

#Obbiettivo: classificare lo stato preictal e interictal

#stacco chb01 34 e 36

import read_data
#import make_dataset as mk
#import os
#from pathlib import Path
import mne
import pandas as pd
import eeg_recording as eeg
from datetime import timedelta as Timedelta


def print_eeg_info():
    patients = read_data.read_data()
    dataset_duration = 0
    for p in patients:
        print('Patient ID:', p.id)
        recordings = p.recordings
        seizure_time = 0
        total_n_seizures = 0
        total_duration = 0
        for r in recordings:
            total_duration += r.duration
            total_n_seizures += r.n_seizures
            for i in range(r.n_seizures):
                seizure_time += r.seizures[i][1] - r.seizures[i][0]
        print(f'Number of seizures: {total_n_seizures} | Total seizure time: {seizure_time} seconds ({seizure_time / 60:.2f} minutes) | Total duration: {total_duration} seconds ({total_duration / 3600:.2f} hours)\n')
        dataset_duration += total_duration
    print(f'Dataset duration: {dataset_duration} seconds ({dataset_duration / 3600:.2f} hours)')

def print_eeg_info2():
    patients = read_data.read_data()
    for p in patients:
        recordings = iter(p.recordings)
        r = next(recordings)
        next_r = next(recordings, None)
        while next_r:
            print(f'{next_r.id} - {r.id}, {next_r.file_start} - {r.file_end} = {(next_r.file_start - r.file_end).total_seconds()}')
            r = next_r
            next_r = next(recordings, None)
        print()
    return True

def print_eeg_info3():
    patients = read_data.read_data()
    for p in patients:
        print('Patient ID:', p.id)
        recordings = iter(p.recordings)
        for r in recordings:
            print(f'{r.id}: {r.file_start} - {r.file_end}')
        print()

def media_distacco():
    patients = read_data.read_data()
    for p in patients:
        print('Patient ID:', p.id)
        recordings = iter(p.recordings)
        r = next(recordings)
        next_r = next(recordings, None)
        sum = 0
        i = 0
        while next_r:
            diff = (next_r.file_start - r.file_end).total_seconds()
            if diff < 300:
                sum += diff
                i += 1
            r = next_r
            next_r = next(recordings, None)
        print(f'Media: {sum / i}')
        print()
    return True

def massimo_distacco():
    patients = read_data.read_data()
    for p in patients:
        print('Patient ID:', p.id)
        recordings = iter(p.recordings)
        r = next(recordings)
        next_r = next(recordings, None)
        sum = 0
        max_diff = 0
        while next_r:
            diff = (next_r.file_start - r.file_end).total_seconds()
            if diff > max_diff:
                id = (r.id, next_r.id)
                max_diff = diff
            r = next_r
            next_r = next(recordings, None)
        print(f'Massimo distacco: {max_diff}', id)
        print()
    return True


def prova(recordings):
    for r in recordings:
        if r.n_seizures > 0:
            print(r.id, r.file_start, r.file_end, r.n_seizures)
            print(r.get_seizure_start()[0])
            #print(r.seizures[0][0])

    return True

def extract_interticat_preictal(patient):
    preictal_interictal = patient.getInterictalPreictalTimes()
    file_start = patient.getRecordingByTime(preictal_interictal[0][0])
    file_end = patient.getRecordingByTime(preictal_interictal[0][1])
    raw = mne.io.read_raw_edf(f'data/{patient.id}/{file_start.id}.edf', verbose='ERROR')
    raw.crop(tmin=(preictal_interictal[0][0] - file_start.file_start).total_seconds(), tmax=14399.9960)
    
    print(raw.n_times)

    # start = (preictal_interictal[0][0] - (patient.getRecordingByTime(preictal_interictal[0][0])).file_start).total_seconds()

    # end = (preictal_interictal[0][1] - (patient.getRecordingByTime(preictal_interictal[0][1])).file_start).total_seconds()
    # patient.getRecordingByTime(preictal_interictal[0][1])
    # start = 
    # end =
    #raw = mne.io.read_raw_edf(f'data/{patient.id}/{patient.recordings[0].id}.edf', verbose='ERROR')

    # seizure_start = patient.getSeizureTimes()
    # print(len (patient.getInterictalPreictalTimes()))

    #TODO:controlla che nelle 5 ore precedenti non ci siano crisi
    #print((seizure_start[2][0] - seizure_start[1][1]).total_seconds())


    #controlla che nelle 5 ore precedenti non ci siano crisi
    # preictal_intericta_start = seizure_start[0][0] - Timedelta(hours=5)

    # print(seizure_start[0][0])
    # print(patient.getRecordingByTime(seizure_start[0][0]).id)
    # print(preictal_intericta_start)
    # print(patient.getRecordingByTime(preictal_intericta_start).id)

    #rec_seizure = patient.getSeizureRecordings()
    #[print(s.id) for s in seizure]

    #print(first_seizure.id)
    #[print(s.id) for s in seizure[1:]]
        


def main():
    patients = read_data.read_data()
    # extract_interticat_preictal(patients[2])
    print(patients[2].get_seizures_datetimes())
    return True

if __name__ == '__main__':
    main()

# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#ictal state     : periodo di tempo durante la crisi.  Durata: variabile
#postictal state : periodo di tempo dopo la crisi.     Durata: da definire
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore