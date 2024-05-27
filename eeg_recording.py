from datetime import datetime

class eeg_recording:
    """
    Class to store the information of an EEG recording.
    """
    def __init__(self, id, channels, file_start, file_end, sampling_rate, n_seizures = 0, seizures = []):
        self.id = id
        self.channels = channels
        self.file_start = file_start
        self.file_end = file_end
        self.duration = (file_end - file_start).total_seconds()
        self.sampling_rate = sampling_rate

        self.n_seizures = n_seizures
        self.seizures = seizures

    def __str__(self):
        return f'{self.id}: {self.file_start} - {self.file_end}, {self.duration} seconds, {self.n_seizures} seizures'