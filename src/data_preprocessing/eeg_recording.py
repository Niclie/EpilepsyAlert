import datetime as dt


class EEGRec:
    """
    Class to store the information of an EEG recording.
    """

    def __init__(self, id, start, end, n_seizures, seizures, channels, sampling_rate):
        """
        Initialize the EEG recording.

        Args:
            id (str): ID of the recording.
            start (datetime): start datetime of the recording.
            end (datetime): end datetime of the recording.
            n_seizures (int): number of seizures in the recording.
            seizures (list): list of tuples with the start and end of the seizures. Each tuple contains the start and end in seconds.
            channels (list): list of channels of the recording.
            sampling_rate (int): sampling rate of the recording.
        """

        self.id = id
        self.start = start
        self.end = end
        self.n_seizures = n_seizures
        self.seizures = seizures
        self.channels = channels
        self.sampling_rate = sampling_rate


    def get_seizures_datetimes(self):
        """
        Get the start and end datetimes of the seizures.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures.
        """

        return [(self.start + dt.timedelta(seconds=start), self.start + dt.timedelta(seconds=end)) for start, end in self.seizures]


    def __str__(self):
        return f'ID: {self.id} Start: {self.start} End: {self.end} Seizures: {self.n_seizures}: {self.seizures}'


    def __repr__(self):
        return self.__str__()