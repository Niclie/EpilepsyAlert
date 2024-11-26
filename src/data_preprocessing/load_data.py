import os
import re
from datetime import datetime, timedelta
from src.utils.constants import *
from src.data_preprocessing.eeg_recording import EEGRec
from src.data_preprocessing.patient import Patient
from src.data_preprocessing.seizure import Seizure
from mne.io import read_raw_edf


def convert_time(time_str, reference_time=None):
    """
    Convert the time string to a datetime object considering the last date for determining if the time is from the next day.

    Args:
        time_str (str): time string to convert.
        reference_time (datetime): last known date for determining if the time is from the next day.

    Returns:
        datetime: datetime object with the converted time.
    """
    reference_time = reference_time or datetime.strptime('00:00:00', TIME_FORMAT)
    hour, minute, second = map(int, time_str.split(':'))
    
    converted_date = reference_time.replace(hour=hour%24, minute=minute, second=second)
    
    return converted_date if converted_date.time() >= reference_time.time() else converted_date + timedelta(days=1)


def load_summary_from_file(p_id, path=None):
    """
    Load the summary of the EEG recordings from the specified file. The file should be named as {p_id}-summary.txt.

    Args:
        p_id (str): patient ID.
        path (str, optional): folder where the data is stored. Defaults to DATA_FOLDER.

    Returns:
        Patient: patient with the EEG recordings.
    """
    path = path or f'{DATA_FOLDER}/{p_id}/{p_id}-summary.txt'
    with open(path, 'r') as file:
        sampling_rate = file.readline().split()[3]
        content = file.read()
    
    files = re.findall(REGEX_BASE_INFO_SELECTOR, content)
    seizure_details = re.findall(REGEX_SEIZURE_INFO_SELECTOR, content)
    
    last_date = datetime.strptime('00:00:00', TIME_FORMAT)
    s_count = 0
    rec_info = []
    for rec_id, rec_start, rec_end, n_seizures in files:
        n_seizures = int(n_seizures)
        rec_start, rec_end = map(lambda t: convert_time(t, last_date), (rec_start, rec_end))
        seizures = [
            Seizure(
                f'{rec_id}_{i}', 
                rec_start + timedelta(seconds=int(s_start)), 
                rec_start + timedelta(seconds=int(s_end))
            ) 
            for i, (s_start, s_end) in enumerate(seizure_details[s_count:s_count + n_seizures], 1)
        ]
        
        rec_info.append(EEGRec(rec_id, rec_start, rec_end, seizures, int(sampling_rate)))
        
        s_count += n_seizures
        last_date = rec_end
        
    return Patient(p_id, rec_info)


def load_summaries_from_folder(path=None, exclude=None):
    """
    Load the summaries of the EEG recordings from the specified folder. The files should be named as {p_id}-summary.txt.

    Args:
        data_folder (str, optional): folder where the data is stored. Defaults to DATA_FOLDER.
        exclude (list, optional): list of patient IDs to exclude. Defaults to None.

    Returns:
        list: list of patients with their EEG recordings.
    """

    path = path or DATA_FOLDER
    p_ids = next(os.walk(path))[1]

    return [load_summary_from_file(id) for id in p_ids if id not in exclude]


def load_eeg_data(path, start_seconds=0, end_seconds=None):
    """
    Retrieve EEG data from a specified file path.
    
    Args:
        path (str, optional): the file path to retrieve the data from.
        start_seconds (int, optional): starting time in seconds to retrieve the data from. Defaults to 0.
        end_seconds (int, optional): ending time in seconds to retrieve the data from. Defaults to None.

    Returns:
        numpy.ndarray: EEG data.
    """
    
    raw = read_raw_edf(path, verbose='ERROR', preload=True)
    raw.pick(CHANNELS)
    data = raw.get_data(tmin=start_seconds, tmax=end_seconds, units='uV').T
    raw.close()
    
    return data