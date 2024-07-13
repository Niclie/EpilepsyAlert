#This file contains all the constants used in the project.
ONE_SECOND_DATA             = 256

DATA_FOLDER                 = 'data/raw'
DATASET_FOLDER              = 'data/processed'

TIME_FORMAT                 = '%H:%M:%S'

REGEX_FILE_INFO_PATTERN     = r'File Name.*\nFile Start.*\nFile End.*\nNumber of Seizures.*\n(?:Seizure (?:\d+ )?Start.*\nSeizure (?:\d+ )?End.*\n?)*'
REGEX_CHANNEL_SELECTOR      = r'Channel \d*:\s*(.*)'
REGEX_BASE_INFO_SELECTOR    = r'File Name:\s*(.*).edf\nFile Start Time:\s*(.*)\nFile End Time:\s*(.*)\nNumber of Seizures in File:\s*(\d+)'
REGEX_SEIZURE_INFO_SELECTOR = r'Seizure (?:\d+ )?Start Time:\s*(\d+) seconds\nSeizure (?:\d+ )?End Time:\s*(\d+) seconds'