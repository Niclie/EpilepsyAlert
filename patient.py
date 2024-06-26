import datetime


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
    

    def get_seizure_recordings(self):
        """
        Get the recordings with seizures.

        Returns:
            list: list of recordings with seizures.
        """

        return [rec for rec in self.recordings if rec.n_seizures > 0]
    

    def get_recording_by_ID(self, id):
        """
        Get the recording with the specified ID.

        Args:
            id (str): ID of the recording.

        Returns:
            EEGRec: recording with the specified ID.
        """

        return next((rec for rec in self.recordings if rec.id == id), None)
    

    def get_seizures_datetimes(self):
        """
        Get the start and end datetimes of the seizures.

        Returns:
        #TODO
        """

        rec_reizures = self.get_seizure_recordings()
        return [rec.get_seizures_datetimes() for rec in rec_reizures]


    def get_recording_by_datetime(self, time):
        """
        Get the recording that contains the specified time.

        Args:
            time (datetime): time to get the recording.

        Returns:
            EEGRec: recording that contains the specified time.
        """

        for r in self.recordings:
            if r.file_start <= time <= r.file_end:
                return r
        return None
    

    def get_interictal_preictal_datetimes(self):
        """
        Get the interictal and preictal datetimes.

        Returns:
            list: list of tuples with the start and end datetimes of the interictal and preictal periods. The interictal period is at least 4 hours long before a preictal period, which is 1 hour long before a seizure.
        """

        interictal_preictal = []
        for r in self.get_seizure_recordings():
            for s in r.seizures:
                end = r.start + datetime.timedelta(seconds = s[0] - 1)
                start = end - datetime.timedelta(hours = 5)
                #check if there are seizures in the range between start and end
                for rec_seizure in self.get_seizures_datetimes():
                    if not any(start <= sd[0] <= end for sd in rec_seizure):
                        interictal_preictal.append((start, end))
                    # for seizure in rec_seizure:
                    #     if not (start <= seizure[0] <= end):
                    #         interictal_preictal.append((start, end))

                # if not any(start <= sd[0] <= end for sd in self.get_seizures_datetimes()):
                #     interictal_preictal.append((start, end))
        
        return interictal_preictal
    

    def __str__(self):
        return f'{self.id}: {len(self.recordings)} recordings'
    

    def __repr__(self):
        return self.__str__()

         
    # def checkSeizure(self, time):
    #     """
    #     Check if the time is within a seizure.
    #     """
    #     for r in self.recordings:
    #         for seizure in r.seizures:
    #             if seizure[0] <= time <= seizure[1]:
    #                 return True
    #     return False