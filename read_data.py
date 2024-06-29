import os
from datetime import datetime, timedelta, time
from eeg_recording import EEGRec
from patient import Patient as patient
import re

DATA_FOLDER = 'data'
TIME_FORMAT = '%H:%M:%S'

def convert_time(time_str, days_, last_date, time_format=TIME_FORMAT):


    hour = int(time_str.split(':')[0])
    if hour >= 24:
        hour = f'{(hour - 24):02d}'
        date = datetime.strptime(hour + ':' + time_str[3:], time_format)
    else:
        date = datetime.strptime(time_str, time_format)

    if time(date.hour, date.minute, date.second) < time(last_date.hour, last_date.minute, last_date.second):
        days_ += 1

    return date + timedelta(days=days_), days_


def load_summary_from_file(p_id, data_folder=DATA_FOLDER):
    with open(os.path.join(data_folder, p_id, f'{p_id}-summary.txt'), 'r') as file:
        content = re.split(r'\*+', file.read())

    #read sampling rate    
    sampling_rate = content[0].split()[3]
    
    channels_selector = re.compile(r'Channel \d*:\s*(.*)')

    file_info_pattern = re.compile(r'File Name.*\nFile Start.*\nFile End.*\nNumber of Seizures.*\n(?:Seizure (?:\d+ )?Start.*\nSeizure (?:\d+ )?End.*\n?)*')
    base_info_selector = re.compile(r'File Name:\s*(.*).edf\nFile Start Time:\s*(.*)\nFile End Time:\s*(.*)\nNumber of Seizures in File:\s*(\d+)')
    seizure_selector = re.compile(r'Seizure (?:\d+ )?Start Time:\s*(\d+) seconds\nSeizure (?:\d+ )?End Time:\s*(\d+) seconds')

    fileInfoList = [
        EEGRec(
            *base_info_selector.search(file_info).group(1, 2, 3, 4),
            [(start, end) for (start, end) in seizure_selector.findall(file_info)],
            channels_selector.findall(split),
            sampling_rate
        )
        for split in content[2:] for file_info in file_info_pattern.findall(split)]
    
    return fileInfoList


def load_summaries_from_folder(data_folder=DATA_FOLDER, time_format=TIME_FORMAT):
    """
    For each patient in data_folder, read the summary file and create a patient object.

    Args:
        data_folder (_type_, optional): folder with the recordings of each patient. Defaults to DATA_FOLDER.
        time_format (_type_, optional): format of the time. Defaults to TIME_FORMAT.

    Returns:
        list: list of patient objects.
    """
    p_ids = os.listdir(data_folder)

    return [patient(id, load_summary_from_file(id, data_folder)) for id in p_ids]


def main():
    load_summary_from_file('chb09')

if __name__ == '__main__':
    main()