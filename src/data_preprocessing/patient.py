import datetime
import time
import src.utils.constants as constants
import mne
import numpy as np
import pandas as pd
from collections import defaultdict


class Patient:
    """
    Class to represent a patient.
    """

    def __init__(self, id, recordings):
        """
        Initialize the patient.

        Args:
            id (str): id of the patient.
            recordings (list): list of EEG recordings.
        """

        self.id = id
        self.recordings = recordings


    def get_seizure_recordings(self, start_index = 0, end_index = None):
        """
        Get the recordings with seizures. If start_index and end_index are provided, only the recordings between those indexes will be considered.

        Args:
            start_index (int, optional): index of the first recording to consider. Defaults to None.
            end_index (int, optional): index of the last recording to consider. Defaults to None.

        Returns:
            list: list of recordings with seizures.
        """

        return [rec for rec in self.recordings[start_index:end_index] if rec.n_seizures > 0]


    def get_seizures_datetimes(self, start_index = 0, end_index = None):
        """
        Get the datetimes of the seizures. If start_index and end_index are provided, only the recordings between those indexes will be considered.

        Args:
            start_index (int, optional): index of the first recording to consider. Defaults to None.
            end_index (int, optional): index of the last recording to consider. Defaults to None.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures.
        """

        return [seizure_datetime 
                for seizure_rec in self.get_seizure_recordings(start_index, end_index) 
                for seizure_datetime in seizure_rec.get_seizures_datetimes()]


    def get_recording_index_by_datetime(self, target_datetime):
        """
        Get the recording that contains the specified datetime.

        Args:
            target_datetime (datetime): datetime to search.

        Returns:
            EEGRec: recording that contains the specified datetime.
        """

        return next((i for i, rec in enumerate(self.recordings) if rec.start <= target_datetime <= rec.end), None)


    def get_clean_seizure_datetimes(self, start_index = 0, end_index = None, interictal_hour = 4, preictal_hour = 1):
        """
        Get the datetimes of the seizures with at least interictal_hour hours of interictal state and preictal_hour hours of preictal state. If start_index and end_index are provided, only the recordings between those indexes will be considered.

        Args:
            start_index (int, optional): index of the first recording to consider. Defaults to 0.
            end_index (int, optional): index of the last recording to consider. Defaults to None.
            interictal_hour (int, optional): hours of interictal state. Defaults to 4.
            preictal_hour (int, optional): hours of preictal state. Defaults to 1.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures.
        """

        last_end = self.recordings[start_index].start
        interictal_preictal = []
        for sd in self.get_seizures_datetimes(start_index, end_index):
            if sd[0] - last_end >= datetime.timedelta(hours = interictal_hour + preictal_hour):
                interictal_preictal.append(sd)
            last_end = sd[1]

        return interictal_preictal
    
    
    def get_continuous_recording_indexes(self, range_minutes=5, start_index = 0, end_index = None):
        """
        Get a list of tuples with the start and end indexes of the continuous recordings. Two recordings are considered continuous if the difference between their end and start is less than range_minute minutes.

        Args:
            range_minutes (int, optional): minutes of difference between the end of a recording and the start of the next one to consider them continuous. Defaults to 5.

        Returns:
            list: list of tuples with the start and end indexes of the continuous recordings.
        """
        
        if end_index is None:
            end_index = len(self.recordings) - 1

        continuous_recording_indexes = []
        start_segment = start_index
        for i, rec in enumerate(self.recordings[start_index + 1: end_index], start_index + 1):
            if rec.start - self.recordings[i - 1].end > datetime.timedelta(minutes=range_minutes) or rec.channels != self.recordings[i - 1].channels:
                continuous_recording_indexes.append((start_segment, i - 1))
                start_segment = i
        
        if self.recordings[end_index].start - self.recordings[end_index-1].end <= datetime.timedelta(minutes=range_minutes):
            continuous_recording_indexes.append((start_segment, end_index))
        else:
            continuous_recording_indexes.append((start_segment, end_index - 1))
            continuous_recording_indexes.append((end_index, end_index))

        return continuous_recording_indexes


    def make_dataset(self, in_path = constants.DATA_FOLDER, out_path = constants.DATASET_FOLDER):
        """
        Create the dataset from the patient's recordings. The dataset will be saved in the specified folder with the name {patient_id}.parquet. It will contain data from the interictal and preictal states, with the class in the last column (0 for interictal and 1 for preictal). The interictal state will include data from 4 hours before the preictal state, and the preictal state will include data from 1 hour before the seizure. Only the group with the most recordings and the same channels will be considered. This group will be divided into segments where the recordings are continuous. The dataset will be in Parquet format to be read with pandas.

        Args:
            in_path (str, optional): path where the recordings are stored. Defaults to constants.DATA_FOLDER.
            out_path (str, optional): path where the dataset will be saved. Defaults to constants.DATASET_FOLDER.

        Returns:
            bool: true if the dataset was created, False otherwise.
        """

        def retrive_data(start_datetime, end_datetime, verbosity = 'ERROR'):
            """
            Retrieve the data from the recordings between the specified datetimes.

            Args:
                start_datetime (datetime): datetime to start the data retrieval.
                end_datetime (datetime): datetime to end the data retrieval.
                verbosity (str, optional): verbosity of the mne.io.read_raw_edf function. Defaults to 'ERROR'.

            Returns:
                np.array: data from the recordings between the specified datetimes.
            """
            phase_recordings = self.recordings[self.get_recording_index_by_datetime(start_datetime):self.get_recording_index_by_datetime(end_datetime) + 1]
            phase_data = []

            raw = mne.io.read_raw_edf(in_path + f'/{phase_recordings[0].id}.edf', verbose = verbosity, include = phase_recordings[0].channels)
            phase_data.append(raw.get_data(tmin = (start_datetime - phase_recordings[0].start).total_seconds(),
                                           tmax = (end_datetime - phase_recordings[0].start).total_seconds()).T)
            
            if len(phase_recordings) < 2:
                return np.concatenate((phase_data))

            for rec in phase_recordings[1:-1]:
                raw = mne.io.read_raw_edf(in_path + f'/{rec.id}.edf', verbose = 'ERROR', include = rec.channels)
                phase_data.append(raw.get_data().T)
                
            raw = mne.io.read_raw_edf(in_path + f'/{phase_recordings[-1].id}.edf', verbose = 'ERROR', include = phase_recordings[-1].channels)
            phase_data.append(raw.get_data(tmax = (end_datetime - phase_recordings[-1].start).total_seconds()).T)
            
            return np.concatenate((phase_data))

        if in_path == constants.DATA_FOLDER:
            in_path += f'/{self.id}'

        grouped_recordings = defaultdict(list)
        for i, rec in enumerate(self.recordings):
            channels_key = tuple(sorted(rec.channels))
            grouped_recordings[channels_key].append(i)

        max_group = max(grouped_recordings.values(), key=len)
        start_rec, end_rec = max_group[0], max_group[-1]
        recordings = self.recordings[start_rec:end_rec + 1]
        
        print(f'Creating dataset for {self.id}...')

        data = []
        class_index = len(recordings[0].channels)
        gap = self.get_max_gap_within_threshold()
        for tuple_index in self.get_continuous_recording_indexes(range_minutes = gap, start_index = start_rec, end_index = end_rec):
            for sd in self.get_clean_seizure_datetimes(start_index = tuple_index[0], end_index = tuple_index[1],):
                end_preictal = sd[0] - datetime.timedelta(seconds = 1)
                start_preictal = end_preictal - datetime.timedelta(hours = 1)
                end_interictal = start_preictal
                start_interictal = end_preictal - datetime.timedelta(hours = 4)

                interictal_data = retrive_data(start_interictal, end_interictal)
                data.append(np.insert(interictal_data, class_index, 0, axis=1))

                preictal_data = retrive_data(start_preictal, end_preictal)
                data.append(np.insert(preictal_data, class_index, 1, axis=1))
        
        if data == []:
            print(f'No dataset created for {self.id}\n')
            return False

        start_time = time.time()
        df = pd.DataFrame(np.concatenate((data)))
        print(f'Creating file {self.id}.parquet...')
        df.to_parquet(f'{out_path}/{self.id}.parquet')
        print(f'File created in {time.time() - start_time:.2f} seconds\n')

        return True
    

    def get_max_gap_within_threshold(self, max_diff_minutes = 10):
        """
        Get the maximum gap between two recordings that is less than the specified threshold.

        Args:
            max_diff_minutes (int, optional): maximum difference in minutes between two recordings to consider them continuous. Defaults to 10.

        Returns:
            int: maximum gap between two recordings that is less than the specified threshold.
        """
        diff = [(self.recordings[i + 1].start - rec.end).total_seconds() / 60 for i, rec in enumerate(self.recordings[:-1])]
        filtered_diff = list(filter(lambda x: x < max_diff_minutes, diff))

        return max(filtered_diff)


    def __str__(self):
        return f'{self.id}: {len(self.recordings)} recordings'
    

    def __repr__(self):
        return self.__str__()