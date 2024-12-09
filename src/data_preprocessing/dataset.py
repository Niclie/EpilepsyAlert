from src.utils.constants import *
from src.data_preprocessing.load_data import load_eeg_data
from datetime import timedelta
import numpy as np
import time
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Class to create a dataset from a patient's recordings.
    """
    
    def __init__(self, patient, in_path=None, interictal_hour=4, preictal_hour=1, segment_size=5, balance=True, split=True, save=True, out_path=None):
        """
        Initialize the dataset class.

        Args:
            patient (Patient): The patient object containing recordings and metadata.
            in_path (str, optional): Path to the input data directory. Defaults to None.
            interictal_hour (int, optional): Number of hours to consider for the interictal phase. Defaults to 4.
            preictal_hour (int, optional): Number of hours to consider for the preictal phase. Defaults to 1.
            segment_size (int, optional): Size of the segments in seconds. Defaults to 5.
            balance (bool, optional): Whether to balance the dataset. Defaults to True.
            split (bool, optional): Whether to split the dataset into train and test sets. Defaults to True.
            save (bool, optional): Whether to save the dataset to a file. Defaults to True.
            out_path (str, optional): Path to the output data directory. Defaults to None.
        """
        self.patient = patient
        self.in_path = in_path or f'{DATA_FOLDER}/{patient.id}'
        self.interictal_hour = interictal_hour
        self.preictal_hour = preictal_hour
        self.segment_size = segment_size
        self.balance = balance
        self.split = split
        self.save = save
        self.out_path = out_path or f'{DATASETS_FOLDER}'
        
        self.recordings = patient.recordings
        self.sampling_rate = self.recordings[0].sampling_rate
        
        
    def make_dataset(self):
        """
        Create the dataset from the patient's recordings.

        Returns:
            dict: Dictionary containing the dataset.
        """
        prev_seizure_end = self.recordings[0].start        
        dataset_preictal = []
        dataset_interictal = []
        
        for i, rec in enumerate(self.recordings):
            for seizure in rec.seizures:
                try:
                    start_preictal, end_preictal, next_ref = self.__get_phase_datetimes(seizure.start, rec_index=i)
                    start_interictal, end_interictal = self.__get_phase_datetimes(start_preictal, next_ref, 'interictal')
                    
                    if not(start_interictal <= prev_seizure_end <= end_preictal):                        
                        data_preictal = self.__retrive_data(start_preictal, end_preictal, i)
                        data_interictal = self.__retrive_data(start_interictal, end_interictal, next_ref)
                        
                        data_preictal = self.__segment_data(data_preictal)
                        data_interictal = self.__segment_data(data_interictal)
                        
                        dataset_preictal.extend(data_preictal)
                        dataset_interictal.extend(data_interictal)
                except ValueError as err:
                    print(f'{seizure.id}: {err}')
                    
                prev_seizure_end = seizure.end
        
        if len(dataset_preictal) + len(dataset_interictal) < 1:
            print(f'No dataset created for {self.patient.id}\n')
            return None
        
        if self.balance:
            dataset_interictal = self.__balance_dataset(dataset_interictal, dataset_preictal)
                
        if self.split:
            train_data, train_labels, test_data, test_labels = self.__split_data(dataset_interictal, dataset_preictal)
            data = {'train_data': train_data, 
                    'train_labels': train_labels, 
                    'test_data': test_data, 
                    'test_labels': test_labels
            }
        else:
            dataset = np.concatenate([dataset_interictal, dataset_preictal])
            labels = np.concatenate([np.zeros(len(dataset_interictal)), np.ones(len(dataset_preictal))])
            
            data = {'data': dataset,
                    'labels': labels
            }
        
        if self.save:
            start_time = time.time()
            print(f'Creating file for {self.patient.id}...')

            filename = f'{self.out_path}/{self.patient.id}.npz'
            np.savez(filename, **data)
            
            print(f'File created in {time.time() - start_time:.2f} seconds\n')

        return data
    
    def __standardize_eeg(self, data, batch_size=100):
        """
        Standardizes EEG data for each channel.
        This method standardizes the EEG data by subtracting the mean and dividing by the standard deviation for each window and channel.
        The standardization is performed in batches to handle large datasets efficiently.
        Args:
            data (numpy array): EEG data of shape (n_windows, n_samples, n_channels).
            batch_size (int, optional): Batch size for standardization. Defaults to 100.
        Returns:
            numpy array: Standardized data with the same shape as the input.
        """
        standardized_data = np.zeros_like(data)
        for start in range(0, len(data), batch_size):
            end = start + batch_size
            batch = data[start:end]
            mean = batch.mean(axis=1, keepdims=True)  # Media per finestra e canale
            std = batch.std(axis=1, keepdims=True)    # Deviazione standard per finestra e canale
            standardized_data[start:end] = (batch - mean) / (std + 1e-8)  # Evita divisione per zero
        
        return standardized_data
    
    
    def __split_data(self, dataset_interictal, dataset_preictal, train_size=0.8, standardize=True):
        """
        Split the dataset into train and test sets.

        Args:
            dataset_interictal (list): List of interictal data.
            dataset_preictal (list): List of preictal data.
            train_size (float, optional): Size of the train set. Defaults to 0.8.

        Returns:
            tuple: Tuple containing the train and test sets and their labels.
        """
        n_interictal = len(dataset_interictal)
        n_preictal = len(dataset_preictal)
        
        data = np.concatenate([dataset_interictal, dataset_preictal])
        
        if standardize: data = self.__standardize_eeg(data)
        
        labels = np.concatenate([np.zeros(n_interictal), np.ones(n_preictal)])
        
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, train_size=train_size, stratify=labels, random_state=35
        )
        
        return train_data, train_labels, test_data, test_labels
    
    
    def __balance_dataset(self, dataset_interictal, dataset_preictal):
        """
        Balance the dataset by selecting a random sample of the interictal data.

        Args:
            dataset_interictal (list): List of interictal data.
            dataset_preictal (list): List of preictal data.

        Returns:
            list: List of balanced interictal data.
        """
        random_index = np.random.choice(len(dataset_interictal), len(dataset_preictal), replace=False)
        balanced_dataset_interictal = [dataset_interictal[i] for i in random_index]
        
        return balanced_dataset_interictal
    
        
    def __segment_data(self, data):
        """
        Segment the data into smaller segments.

        Args:
            data (np.array): The data to segment.

        Returns:
            np.array: The segmented data.
        """
        n_samples = self.sampling_rate * self.segment_size
        n_segments = len(data) // n_samples

        return np.array([data[i * n_samples:(i + 1) * n_samples] for i in range(n_segments)])
    
    
    def __retrive_data(self, start_datetime, end_datetime, rec_index):
        """
        Retrieve the data from the recordings

        Args:
            start_datetime (datetime): The start datetime of the data.
            end_datetime (datetime): The end datetime of the data.
            rec_index (int): The index of the recording from which to retrieve the data.

        Returns:
            np.array: The retrieved data.
        """
        
        data = None
        for i in range(rec_index, - 1, -1):
            if self.recordings[i].start <= end_datetime <= self.recordings[i].end:
                end_seconds = (end_datetime - self.recordings[i].start).total_seconds()
                if self.recordings[i].start <= start_datetime <= self.recordings[i].end:
                    return load_eeg_data(f'{self.in_path}/{self.recordings[i].id}.edf', (start_datetime - self.recordings[i].start).total_seconds(), end_seconds)
            
                data = load_eeg_data(f'{self.in_path}/{self.recordings[i].id}.edf', 0, end_seconds)
                
            elif data is not None:
                if self.recordings[i].start <= start_datetime <= self.recordings[i].end:
                    rec_data = load_eeg_data(f'{self.in_path}/{self.recordings[i].id}.edf', (start_datetime - self.recordings[i].start).total_seconds())
                    data = np.vstack((data, rec_data))
                    return data[::-1]
                
                rec_data = load_eeg_data(f'{self.in_path}/{self.recordings[i].id}.edf')
                data = np.vstack((data, rec_data))
    
    
    def __get_phase_datetimes(self, datetime_ref, rec_index, phase_type='preictal', gap=1):
        """
        Get the start and end datetimes of the preictal or interictal phase.

        Args:
            datetime_ref (datetime): The reference datetime from which to calculate the phase datetimes.
            rec_index (int): The index of the recording from which to calculate the phase datetimes.
            phase_type (str, optional): The type of phase to calculate. Defaults to 'preictal'.
            gap (int, optional): The gap in seconds to consider. Defaults to 1.

        Raises:
            ValueError: If there is not enough recording time for the interictal phase.

        Returns:
            tuple: Tuple containing the start, end and next reference datetimes.
        """
        end, rec_index = self.__check_datetime(datetime_ref - timedelta(seconds=gap), rec_index, False)
        
        if phase_type == 'preictal':
            diff = (datetime_ref - timedelta(seconds=gap) - end).total_seconds()
            start, next_ref = self.__check_datetime(end - timedelta(hours=self.preictal_hour), rec_index, True)
            start += timedelta(seconds=diff)
            return start, end, next_ref
        
        length_remaining = self.interictal_hour - (end - self.recordings[rec_index].start).total_seconds() / 3600
        if length_remaining > 0:
            for i in range(rec_index - 1, - 1, -1):
                length_remaining -= (self.recordings[i].end - self.recordings[i].start).total_seconds() / 3600
                if length_remaining <= 0:
                    break
                
            if length_remaining > 0:
                raise ValueError(f'{datetime_ref} not enough recording time for interictal phase of {self.interictal_hour} hours')
                    
        start = self.recordings[i].start + timedelta(hours=abs(length_remaining))
        #start = self.__check_datetime(end - timedelta(hours=self.interictal_hour), rec_index, True)[0]

        return start, end
    
    
    def __check_datetime(self, datetime, rec_index, use_start=True):
        """
        Check if the datetime is within the range of the recordings.

        Args:
            datetime (datetime): The datetime to check.
            rec_index (int): The index of the recording from which to check.
            use_start (bool, optional): Whether to use the start or end of the recording. Defaults to True.

        Raises:
            ValueError: If the datetime is out of the range of available recordings.

        Returns:
            tuple: Tuple containing the datetime and the recording index.
        """
        for i in range(rec_index, - 1, -1):
            if self.recordings[i].start <= datetime <= self.recordings[i].end:
                return (datetime, i)
            
            if i > 0 and self.recordings[i - 1].end <= datetime <= self.recordings[i].start:
                return (self.recordings[i].start, i) if use_start else (self.recordings[i - 1].end, i - 1)
        
        raise ValueError(f'{datetime} is out of the range of available recordings.')
            
            
    def __str__(self) -> str:
        return f'{self.patient.id} dataset with {len(self.recordings)} recordings.'