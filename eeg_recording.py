from datetime import datetime, timedelta

class EEGRec:
    """
    Class to store the information of an EEG recording.
    """
    def __init__(self, id, channels, start, end, sampling_rate, n_seizures = 0, seizures = []):
        """
        Initialize the EEG recording.

        Args:
            id (str): id of the recording.
            channels (list): list of channels of the recording.
            start (datetime): start datetime of the recording.
            end (datetime): end datetime of the recording.
            sampling_rate (int): sampling rate of the recording.
            n_seizures (int, optional): number of seizures. Defaults to 0.
            seizures (list, optional): list of tuples with the start and end of the seizures. Defaults to [].
        """
        self.id = id
        self.channels = channels
        self.start = start
        self.end = end
        self.sampling_rate = sampling_rate
        self.n_seizures = n_seizures
        self.seizures = seizures

    def get_duration(self):
        """
        Get the duration of the recording.

        Returns:
            float: duration of the recording in seconds.
        """
        return (self.end - self.start).total_seconds()

    def get_seizures_datetimes(self):
        """
        Get the start and end datetimes of the seizures.

        Returns:
            list: list of tuples with the start and end datetimes of the seizures.
        """
        return [(self.start + timedelta(seconds=start), self.start + timedelta(seconds=end)) for start, end in self.seizures]

    def __str__(self):
        return f'ID: {self.id} Start: {self.start} End: {self.end} Seizures: {self.n_seizures}'

    def __repr__(self):
        return self.__str__()