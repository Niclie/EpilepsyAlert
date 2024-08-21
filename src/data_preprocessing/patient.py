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


    def group_by_channels(self):
        """
        Groups the recordings by the channels.

        Returns:
            dict: dictionary where the keys are the channels and the values are lists of recording indexes.
        """

        recs = defaultdict(list)
        for i, r in enumerate(self.recordings):
            channels = tuple(r.channels)
            recs[channels].append(i)

        return recs


    def __str__(self):
        """
        String representation of the patient.

        Returns:
            str: string representation of the patient.
        """

        return f'{self.id}: {len(self.recordings)} recordings'
