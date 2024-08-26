import sys
import os
sys.path.append(os.path.abspath('.'))
from src.data_preprocessing.load_data import load_summary_from_file, load_summaries_from_folder
import src.data_preprocessing.preprocess as preprocess
import numpy as np
from datetime import timedelta


def generate_recordings_gap(patient):
    f = open('output2.txt', 'a') 
    f.write(f'{patient.id}:\n')

    recordings = preprocess.max_channel_recordings(patient)[0]

    gap = []
    for recs in recordings:
        for i, r in enumerate(recs[1:], 1):
            gap_val = (r.start - recs[i-1].end).total_seconds()
            gap.append(gap_val)
            f.write(f'{recs[i-1].id} -> {r.id} = {gap_val}\n')
    
    f.write(f'\nMean gap: {np.mean(gap)}')
    f.write("\n\n\n")
    f.close()


def generate_groups(patient):
    recordings = preprocess.max_channel_recordings(patient)[0]
    #recordings = preprocess.get_consecutive_recordings(recordings, range_minutes=preprocess.get_max_gap_within_threshold(recordings))

    f = open('output2.txt', 'a') 
    f.write(f'{patient.id}:\n')

    for recs in recordings:
        f.write(f'{recs[0].id} -> {recs[-1].id}\n')

    f.write("\n\n\n")
    f.close()

def seizure_preictal_interictal_duration(patient, a):
    recordings = preprocess.max_channel_recordings(patient)
    gap_size = recordings[1]
    recordings = recordings[0]
    #recordings = preprocess.get_consecutive_recordings(recordings, range_minutes=preprocess.get_max_gap_within_threshold(recordings))
    f = open('output2.txt', 'a')
    f.write(f'{patient.id}:\n')

    count = 0
    d = []
    for recs in recordings:
        f.write(f'Block: {recs[0].id} -> {recs[-1].id}\n')
        last_seizure_end = recs[0].start
        for r in recs:
            for s in r.get_seizures_datetimes():
                val = round((s[0] - last_seizure_end).total_seconds() / 3600, 3)
                if val >= a:
                    count += 1
                d.append(val)
                f.write(f'Seizure start {r.id} at {s[0]} =\t{val}\n')
                last_seizure_end = s[1]
        f.write('\n')

    f.write(f'\nMean: {np.mean(d)}\n')
    f.write(f'>{a}: {count}')
    f.write("\n\n\n")
    f.close()
    return count


def group_recordings_by_channels(patient):
    f = open('output2.txt', 'a') 
    f.write(f'{patient.id}:\n')

    recordings = patient.group_by_channels()
    
    for recs in recordings.values():
        f.write(f'Channels: {patient.recordings[recs[0]].channels}\n')
        for r in recs:
            f.write(f'{patient.recordings[r].id}\n')

    f.close()
    return

def generate_gaps(patients):
    f = open('output2.txt', 'a') 
    for p in patients:
        f.write(f'{p.id}: {preprocess.max_channel_recordings(p)[1]}\n')

    f.close()

    return

def generate_seizures_datetimes(patient):
    f = open('output2.txt', 'a') 
    f.write(f'{patient.id}\n')
    for r in patient.recordings:
        for s in r.get_seizures_datetimes():
            f.write(f'{r.id}: {s}\n')

    f.close()

    return

def main():
    # patients = load_summaries_from_folder(exclude='chb24')
    # count = 0
    # for p in patients:
    #     a = seizure_preictal_interictal_duration(p, 4)
    #     if a>0:
    #         print(p.id, "\n")
    #         count += 1
    # print(count)
    
    id = 'chb06'
    patient = load_summary_from_file(id)
    #group_recordings_by_channels(patient)
    #generate_recordings_gap(patient)
    #generate_groups(patient)
    generate_seizures_datetimes(patient)

    return

if __name__ == '__main__':
    main()