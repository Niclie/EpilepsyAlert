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

    
    def __str__(self) -> str:
        seizures_str = '\n\t'.join(map(str, self.seizures))
        return f'{self.id}: {self.start} -> {self.end}{'\n\t' + seizures_str if self.seizures else ''}'
