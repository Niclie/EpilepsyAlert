import os
from datetime import datetime, timedelta, time
import eeg_recording
from patient import Patient as patient

DATA_FOLDER = 'data'
TIME_FORMAT = '%H:%M:%S'

def convert_time(time_str, days_, last_date, time_format=TIME_FORMAT):
    """
    Convert a string representing a time to a datetime object.

    Args:
        time_str (string): string representing a time.
        time_format (string, optional): format of the time string. Defaults to TIME_FORMAT.

    Returns:
        datetime: datetime object representing the time.
    """

    hour = int(time_str.split(':')[0])
    if hour >= 24:
        hour = f'{(hour - 24):02d}'
        date = datetime.strptime(hour + ':' + time_str[3:], time_format)
    else:
        date = datetime.strptime(time_str, time_format)

    if time(date.hour, date.minute, date.second) < time(last_date.hour, last_date.minute, last_date.second):
        days_ += 1

    return date + timedelta(days=days_), days_

def read_summary_file(path, patient_id, time_format=TIME_FORMAT):
    """
    Read the summary file of a patient and return a list of eeg_recording objects.

    Args:
        path (string): path to the summary file of the patient.
        patient_id (string): id of the patient.

    Returns:
        list: list of eeg_recording objects.
    """
    recordings = []

    with open(path, 'r') as file:
        lines = file.readlines()

    it = iter(lines)
    # read sampling rate
    sampling_rate = next(it).split()[3]

    # read channels: da modificare! #TODO
    line = next(it)
    while 'Channels' not in line:
        line = next(it)
    next(it)

    channels = []
    for line in it:
        if line == '\n':
            break

        str = line.split()[2]
        
        if str == '.' or str == '-':
            continue
    
        channels.append(str)

    
    # read recording description
    last_date = datetime.strptime('00:00:00', time_format)
    days = 0
    for line in it:
        if 'Channels changed' in line: # check if the channels are changed
            channels = []
            next(it)
            line = next(it)
            while line != '\n':
                channels.append(line.split()[2])
                line = next(it)
            line = next(it)

        # set recording parameters
        id = line.split()[2].split('.')[0]
        start, days = convert_time(next(it).split()[3], days, last_date, time_format)
        last_date = start
        end, days= convert_time(next(it).split()[3], days, last_date, time_format)
        last_date = end

        n_seizures = int(next(it).split()[5])

        # read seizures
        seizures = []
        for _ in range(n_seizures):
            seizure_start = next(it).split()
            if patient_id == 'chb01' or patient_id == 'chb03':
                seizure_start = seizure_start[3]
            else:
                seizure_start = seizure_start[4]

            seizure_end = next(it).split()
            if patient_id == 'chb01' or patient_id == 'chb03':
                seizure_end = seizure_end[3]
            else:
                seizure_end = seizure_end[4]
            seizures.append((int(seizure_start), int(seizure_end)))
        
        recordings.append(eeg_recording.eeg_recording(id, channels, start, end, sampling_rate, n_seizures, seizures))
        next(it, None)
    return recordings


def read_data(data_folder=DATA_FOLDER, time_format=TIME_FORMAT):
    p_ids = os.listdir(data_folder)
    p_summaries = [os.path.join(data_folder, id, f'{id}-summary.txt') for id in p_ids]
    patients = []
    for s in range(len(p_summaries)):
        patients.append(patient(p_ids[s],
                                read_summary_file(p_summaries[s], p_ids[s], time_format)))
    return patients
        
# if __name__ == '__main__':
#     read_data()