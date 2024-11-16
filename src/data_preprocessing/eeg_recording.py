import numpy as np
from mne.io import read_raw_edf
from collections import Counter

class EEGRec:
    """
    Class to store the information of an EEG recording.
    """

    def __init__(self, id, start, end, seizures, sampling_rate):
        """
        Initialize the EEG recording.

        Args:
            id (str): ID of the recording.
            start (datetime): start datetime of the recording.
            end (datetime): end datetime of the recording.
            seizures (list): list of seizures.
            channels (list): list of channels of the recording.
            sampling_rate (int): sampling rate of the recording.
        """

        self.id = id
        self.start = start
        self.end = end
        self.seizures = seizures
        self.sampling_rate = sampling_rate


    def get_seizures_datetimes(self):
        """
        Get the start and end datetimes of the seizures.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures.
        """

        return [(s.start, s.end) for s in self.seizures]
        

    def retrive_data(self, in_path, start_seconds = 0, end_seconds = None, verbosity = 'ERROR', n_channels = 23):
        """
        Retrieve EEG data from a specified file path.
        
        Args:
            in_path (str, optional): the file path to retrieve the data from. Defaults to constants.DATA_FOLDER.
            start_seconds (int, optional): starting time in seconds to retrieve the data from. Defaults to 0.
            end_seconds (int, optional): ending time in seconds to retrieve the data from. Defaults to None.
            verbosity (str, optional): the level of verbosity for the data retrieval process. Defaults to 'ERROR'.
            n_channels (int, optional): the number of channels to retrieve. Defaults to 23.

        Returns:
            numpy.ndarray: retrieved EEG data from the first n_channels.
        """

        raw = read_raw_edf(in_path, verbose=verbosity)
        raw.pick('eeg')
        #raw.filter(1, 40)
        data = raw.get_data(tmin=start_seconds, tmax=end_seconds, picks=self.channels, units='uV').T
        raw.close()
        
        return np.around(data, 4)
        #return data
        
    def __str__(self) -> str:
        seizures_str = '\n\t'.join(map(str, self.seizures))
        return f'{self.id}: {self.start} -> {self.end}{'\n\t' + seizures_str if self.seizures else ''}'
