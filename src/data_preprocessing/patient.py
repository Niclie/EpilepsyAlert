import datetime
import time
import src.utils.constants as constants
import mne
import numpy as np
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


    def make_dataset(self, 
                     in_path              = constants.DATA_FOLDER,
                     segment_size_seconds = 5,
                     balance              = True,
                     split                = True,
                     save                 = True,
                     out_path             = constants.DATASET_FOLDER,
                     compress             = False,
                     verbosity            = 'ERROR'):
        """
        Create a dataset with the interictal and preictal states of the seizures of the patient.

        Args:
            in_path (str, optional): path where the recordings are stored. Defaults to constants.DATA_FOLDER.
            segment_size_seconds (int, optional): size of the segments in seconds if None, the data will not be segmented. Defaults to 5.
            balance (bool, optional): if True, the dataset will be balanced with the same number of interictal and preictal segments. Defaults to True.
            split (bool, optional): if True, the dataset will be split into training and test datasets. Defaults to True..
            save (bool, optional): if True, the dataset will be saved in a .npz file. Defaults to True.
            out_path (str, optional): path where the dataset will be saved. Defaults to constants.DATASET_FOLDER.
            compress (bool, optional): if True, the dataset will be saved in a compressed .npz file. Defaults to False.
            verbosity (str, optional): verbosity of the mne.io.read_raw_edf function. Defaults to 'ERROR'.

        Returns:
            dict: dictionary with the dataset.
        """

        if in_path == constants.DATA_FOLDER:
            in_path += f'/{self.id}'

        grouped_recordings = defaultdict(list)
        for i, rec in enumerate(self.recordings):
            channels_key = tuple(sorted(rec.channels))
            grouped_recordings[channels_key].append(i)

        max_group = max(grouped_recordings.values(), key=len)
        start_rec, end_rec = max_group[0], max_group[-1]
        
        print(f'Creating dataset for {self.id}...')

        n_preictal_segments = []
        label = []
        data = []
        for tuple_index in self.get_continuous_recording_indexes(range_minutes=self.__get_max_gap_within_threshold(), start_index=start_rec, end_index=end_rec):
            for sd in self.get_clean_seizure_datetimes(start_index = tuple_index[0], end_index = tuple_index[1]):
                end_preictal = sd[0] - datetime.timedelta(seconds = 1)
                start_preictal = end_preictal - datetime.timedelta(hours = 1)
                end_interictal = start_preictal
                start_interictal = end_interictal - datetime.timedelta(hours = 4)

                interictal_data = self.__retrive_data(in_path, start_interictal, end_interictal, segment_size_seconds, verbosity)
                interictal_label = np.zeros(interictal_data.shape[0])

                preictal_data = self.__retrive_data(in_path, start_preictal, end_preictal, segment_size_seconds, verbosity)
                preictal_label = np.ones(preictal_data.shape[0])
                n_preictal_segments.append(len(preictal_data))

                data.append(np.concatenate((interictal_data, preictal_data)))
                label.append(np.concatenate((interictal_label, preictal_label)))
        
        if data == []:
            print(f'No dataset created for {self.id}\n')
            return None
        
        if balance:
            rng = np.random.default_rng()
            for i, d in enumerate(data):
                interictal_index = rng.choice(len(d) - n_preictal_segments[i], size=n_preictal_segments[i], replace=False)

                data[i] = np.concatenate((d[interictal_index], d[len(d) - n_preictal_segments[i]:]))
                label[i] = np.concatenate((label[i][interictal_index], 
                                           label[i][len(d) - n_preictal_segments[i]:]))
        
        if split:
            interictal_trainig_indexes = []
            interictal_test_indexes = []
            preictal_training_indexes = []
            preictal_test_indexes = []

            rng = np.random.default_rng()
            for i, d in enumerate(data):
                n_trainig_example = (len(d) * 80) // 100

                interictal_trainig_indexes.append(rng.choice(len(d) - n_preictal_segments[i], size = n_trainig_example // 2, replace=False))
                interictal_test_indexes.append(np.setdiff1d(np.arange(len(d) - n_preictal_segments[i]), interictal_trainig_indexes[i]))

                preictal_training_indexes.append(rng.choice(np.arange(len(d) - n_preictal_segments[i], len(d)), size = n_trainig_example // 2, replace=False))
                preictal_test_indexes.append(np.setdiff1d(np.arange(len(d) - n_preictal_segments[i], len(d)), preictal_training_indexes[i]))
            
            trainig_label = []
            test_label = []

            trainig_data = []
            test_data = []
            for i, d in enumerate(data):
                trainig_data.extend([d[interictal_trainig_indexes[i]], d[preictal_training_indexes[i]]])
                trainig_label.extend([label[i][interictal_trainig_indexes[i]], label[i][preictal_training_indexes[i]]])
                test_data.extend([d[interictal_test_indexes[i]], d[preictal_test_indexes[i]]])
                test_label.extend([label[i][interictal_test_indexes[i]], label[i][preictal_test_indexes[i]]])
            
            data = {'trainig_data': np.concatenate((trainig_data)), 
                    'trainig_label': np.concatenate((trainig_label)), 
                    'test_data': np.concatenate((test_data)), 
                    'test_label': np.concatenate((test_label)), 
                    'channels': self.recordings[start_rec].channels}
        else:
            data = {'data': np.concatenate((data)), 'label': np.concatenate((label)), 'channels': self.recordings[start_rec].channels}


        start_time = time.time()
        print(f'Creating file for {self.id}...')

        if save:
            filename = f'{out_path}/{self.id}.npz'
            if compress:
                np.savez_compressed(filename, **data)
            else:
                np.savez(filename, **data)

        print(f'File created in {time.time() - start_time:.2f} seconds\n')

        return data
    

    def __retrive_data(self, in_path, start_datetime, end_datetime, segment_size_seconds, verbosity):
        """
        Retrieve the data of the patient between the specified datetimes.

        Args:
            in_path (str): path where the recordings are stored.
            start_datetime (datetime): start datetime.
            end_datetime (datetime): end datetime.
            segment_size_seconds (int): size of the segments in seconds.
            verbosity (str): verbosity of the mne.io.read_raw_edf function.

        Returns:
            np.array: data of the patient between the specified datetimes.
        """
        phase_recordings = self.recordings[self.get_recording_index_by_datetime(start_datetime):self.get_recording_index_by_datetime(end_datetime) + 1]
        phase_data = []

        raw = mne.io.read_raw_edf(in_path + f'/{phase_recordings[0].id}.edf', verbose = verbosity, include = phase_recordings[0].channels)
        phase_data.append(raw.get_data(tmin = (start_datetime - phase_recordings[0].start).total_seconds(),
                                       tmax = (end_datetime - phase_recordings[0].start).total_seconds()).T)
        
        if len(phase_recordings) < 2:
            if segment_size_seconds:
                return self.__segment_data(np.concatenate((phase_data)), segment_size_seconds)
            else:
                return np.concatenate((phase_data))

        for rec in phase_recordings[1:-1]:
            raw = mne.io.read_raw_edf(in_path + f'/{rec.id}.edf', verbose = 'ERROR', include = rec.channels)
            phase_data.append(raw.get_data().T)
            
        raw = mne.io.read_raw_edf(in_path + f'/{phase_recordings[-1].id}.edf', verbose = 'ERROR', include = phase_recordings[-1].channels)
        phase_data.append(raw.get_data(tmax = (end_datetime - phase_recordings[-1].start).total_seconds()).T)

        phase_data = np.concatenate((phase_data))

        if segment_size_seconds:
            phase_data = self.__segment_data(phase_data, segment_size_seconds)
        
        return phase_data


    def __segment_data(self, data, segment_size_seconds):
        """
        Segment the data in segments of the specified size. 

        Args:
            data (np.array): data to segment.
            segment_size_seconds (int): size of the segments in seconds.

        Returns:
            np.array: segmented data.
        """
        n_samples = constants.ONE_SECOND_DATA * segment_size_seconds
        n_segments = len(data) // n_samples

        return np.array([data[i * n_samples:(i + 1) * n_samples] for i in range(n_segments)])


    def __get_max_gap_within_threshold(self, max_diff_minutes = 10):
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
        """
        String representation of the patient.

        Returns:
            str: string representation of the patient.
        """
        return f'{self.id}: {len(self.recordings)} recordings'
    

    def __repr__(self):
        """
        Representation of the patient.

        Returns:
            str: representation of the patient.
        """
        return self.__str__()