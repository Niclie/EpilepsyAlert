import datetime
import constants

import mne
import numpy as np


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
            end_index (_type_, optional): index of the last recording to consider. Defaults to None.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures.
        """

        return [seizure_datetime 
                for seizure_rec in self.get_seizure_recordings(start_index, end_index) 
                for seizure_datetime in seizure_rec.get_seizures_datetimes()]


    def get_recording_index_by_datetime(self, time):
        """
        Get the recording that contains the specified datetime.

        Args:
            time (datetime): datetime to search.

        Returns:
            EEGRec: recording that contains the specified datetime.
        """

        return next((i for i, rec in enumerate(self.recordings) if rec.start <= time <= rec.end), None)


    def get_clean_seizure_datetimes(self, start_index = 0, end_index = None, interictal_hour = 4, preictal_hour = 1):
        """
        Get the datetimes of the seizures with at least interictal_hour hours of interictal state and preictal_hour hours of preictal state. If start_index and end_index are provided, only the recordings between those indexes will be considered.

        Args:
            start_index (int, optional): index of the first recording to consider. Defaults to 0.
            end_index (int, optional): index of the last recording to consider. Defaults to None.
            interictal_hour (int, optional): hours of interictal state. Defaults to 4.
            preictal_hour (int, optional): hours of preictal state. Defaults to 1.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures with at least interictal_hour hours of interictal state and preictal_hour hours of preictal state.
        """

        last_end = self.recordings[start_index].start
        interictal_preictal = []
        for sd in self.get_seizures_datetimes(start_index, end_index):
            if sd[0] - last_end >= datetime.timedelta(hours = interictal_hour + preictal_hour):
                interictal_preictal.append(sd)
            last_end = sd[1]

        return interictal_preictal
    
    def get_continuous_recording_indexes(self, range_minute=5):
        """
        Get a list of tuples with the start and end indexes of the continuous recordings. Two recordings are considered continuous if the difference between their end and start is less than range_minute minutes.

        Args:
            range_minute (int, optional): minutes of difference between the end of a recording and the start of the next one to consider them continuous. Defaults to 5.

        Returns:
            list: list of tuples with the start and end indexes of the continuous recordings.
        """
        continuous_recording_indexes = []
        start = 0
        for i, rec in enumerate(self.recordings[1:-1], 1):
            if rec.start - self.recordings[i - 1].end > datetime.timedelta(minutes=range_minute):
                continuous_recording_indexes.append((start, i - 1))
                start = i
        
        if self.recordings[-1].start - self.recordings[-2].end <= datetime.timedelta(minutes=range_minute):
            continuous_recording_indexes.append((start, len(self.recordings) - 1))
        else:
            continuous_recording_indexes.append((start, len(self.recordings) - 2))
            continuous_recording_indexes.append((len(self.recordings) - 1, len(self.recordings) - 1))

        return continuous_recording_indexes


    def make_dataset(self, in_path = constants.DATA_FOLDER, out_path = None):
        in_path += f'/{self.id}'
        
        continuous_recording_indexes = []
        data = []
        for tuple_index in self.get_continuous_recording_indexes():
            seizure_datetimes = self.get_clean_seizure_datetimes(start_index = tuple_index[0], end_index = tuple_index[1])
            for sd in seizure_datetimes:
                end_preictal = sd[0] - datetime.timedelta(seconds = 1)
                start_interictal = end_preictal - datetime.timedelta(hours = 5)

                recs = self.recordings[self.get_recording_index_by_datetime(start_interictal):self.get_recording_index_by_datetime(end_preictal) + 1]
                # sum = 0
                # for i, r in enumerate(recs[:-1]):
                #     print(recs[i+1].id, " - ", r.id, " = ", recs[i + 1].start, " - ", r.end, " = ", (recs[i + 1].start - r.end).total_seconds())
                #     sum += (recs[i + 1].start - r.end).total_seconds()
                # print(sum)
                # return[]
                #exit()
                start = start_interictal
                end = recs[0].end
                for i, r in enumerate(recs[:-1]):
                    raw = mne.io.read_raw_edf(in_path + f'/{r.id}.edf', verbose = 'ERROR')
                    data.append(raw.get_data(tmin = (start - r.start).total_seconds(),
                                             tmax = (end - r.start).total_seconds()).T)
                    start = recs[i + 1].start
                    end = recs[i + 1].end

                data.append(raw.get_data(tmin = (start - recs[-1].start).total_seconds(),
                                         tmax = (end_preictal - recs[-1].start).total_seconds()).T)
                
        return np.concatenate((data))
                    
                    


        return continuous_recording_indexes
    # def get_best_recording_blocks(self):
    #     blocks = self.get_continuous_recording_blocks()

    #     return [b for b in blocks if len(self.get_interictal_preictal_datetimes(b)) > 0]
    

    # def make_dataset(self, in_path = constants.DATA_FOLDER, out_path = None):
        
    #     for b in self.get_best_recording_blocks():
    #         interictal_preictal_datetimes = self.get_interictal_preictal_datetimes(b)
    #         for datetime in interictal_preictal_datetimes:
    #             first_rec = self.get_recording_by_datetime(datetime[0])
    #             last_rec = self.get_recording_by_datetime(datetime[1])

    #             between_recs = self.recordings[self.recordings.index(first_rec):self.recordings.index(last_rec) + 1]

    #             dataset = []
    #             if len(between_recs) == 1:
    #                 raw = mne.io.read_raw_edf(in_path + f'/{self.id}' + f'/{between_recs[0].id}.edf')
    #                 data = raw.get_data(tmin = (datetime[0] - between_recs[0].start).total_seconds(), 
    #                                     tmax = (between_recs[0].end - datetime[1]).total_seconds()).T
    #                 #return data
    #             elif len(between_recs) == 2:
    #                 raw = mne.io.read_raw_edf(in_path + f'/{self.id}' + f'/{between_recs[0].id}.edf')
    #                 data = raw.get_data(tmin = (datetime[0] - between_recs[0].start).total_seconds()).T
    #                 dataset.append(data)
    #                 raw = mne.io.read_raw_edf(in_path + f'/{self.id}' + f'/{between_recs[1].id}.edf')
    #                 data = raw.get_data(tmax = (datetime[1] - between_recs[1].start).total_seconds()).T
    #                 dataset.append(data)
    #                 #return data
    #             elif len(between_recs) >= 3:
    #                 raw = mne.io.read_raw_edf(in_path + f'/{self.id}' + f'/{between_recs[0].id}.edf')
    #                 data = raw.get_data(tmin = (datetime[0] - between_recs[0].start).total_seconds()).T
    #                 dataset.append(data)
    #                 for rec in between_recs[1:-1]:
    #                     raw = mne.io.read_raw_edf(in_path + f'/{self.id}' + f'/{rec.id}.edf')
    #                     data = raw.get_data().T
    #                     dataset.append(data)

    #                 raw = mne.io.read_raw_edf(in_path + f'/{self.id}' + f'/{between_recs[-1].id}.edf')
    #                 data = raw.get_data(tmax = (datetime[1] - between_recs[-1].start).total_seconds()).T
    #                 dataset.append(data)
    #                 #return data
    #     np.savetxt('your_csv_file.csv', dataset, delimiter=',', header=self.recordings[0].channels)

    def __str__(self):
        return f'{self.id}: {len(self.recordings)} recordings'
    

    def __repr__(self):
        return self.__str__()