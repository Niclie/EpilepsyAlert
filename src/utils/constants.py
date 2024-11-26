#This file contains all the constants used in the project.

PATIENTS = ('chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19','chb20', 'chb21', 'chb22', 'chb23')
CHANNELS = ('FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8')

DATA_FOLDER                 = 'data/raw'
DATASETS_FOLDER             = 'data/converted'
PREPROCESSED_FOLDER         = 'data/preprocessed'

RESULTS_FOLDER              = 'results'
MODELS_FOLDER               = 'results/models'
PLOTS_FOLDER                = 'results/plots'

TIME_FORMAT                 = '%H:%M:%S'

REGEX_BASE_INFO_SELECTOR    = r'File Name:\s*(.*).edf\nFile Start Time:\s*(.*)\nFile End Time:\s*(.*)\nNumber of Seizures in File:\s*(\d+)'
REGEX_SEIZURE_INFO_SELECTOR = r'Seizure (?:\d+ )?Start Time:\s*(\d+) seconds\nSeizure (?:\d+ )?End Time:\s*(\d+) seconds'