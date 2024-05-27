class patient:
    """
    Class to represent a patient.
    """
    def __init__(self, id, recordings):
        self.id = id
        self.recordings = recordings

    def __str__(self):
        return f'{self.id}: {len(self.recordings)} recordings'