from datetime import timedelta
import numpy as np
import src.utils.constants as constants
import time


def make_dataset(patient, in_path = constants.DATA_FOLDER, segment_size = 5, balance = True, split = True, save = True, out_path = constants.DATASETS_FOLDER):
    """
    Create a dataset with the interictal and preictal states of the seizures of the patient.

    Args:
        patient (Patient): patient object to create the dataset.
        in_path (str, optional): path where the recordings are stored. Defaults to constants.DATA_FOLDER.
        segment_size (int, optional): size of the segments in seconds if None, the data will not be segmented. Defaults to 5.
        balance (bool, optional): if True, the dataset will be balanced with the same number of interictal and preictal segments. Defaults to True.
        split (bool, optional): if True, the dataset will be split into training and test datasets. Defaults to True.
        save (bool, optional): if True, the dataset will be saved in a .npz file. Defaults to True.
        out_path (str, optional): path where the dataset will be saved. Defaults to constants.DATASETS_FOLDER.

    Returns:
        dict: dictionary with the dataset.
    """

    if in_path == constants.DATA_FOLDER:
        in_path += f'/{patient.id}'
    
    grouped_recordings, gap_size = max_channel_recordings(patient)
    interictal_hour = round((gap_size * 80)/100, 2)
    preictal_hour = gap_size - interictal_hour

    print(f'Creating dataset for {patient.id}...')
    
    data = []
    for recordings in grouped_recordings:
        last_seizure_end = recordings[0].start
        for rec in recordings:
            for s in rec.get_seizures_datetimes():
                if s[0] - last_seizure_end >= timedelta(hours = gap_size):
                    start_preictal, end_preictal = get_phase_datetimes(recordings, s[0], preictal_hour, 1)
                    start_interictal, end_interictal = get_phase_datetimes(recordings, start_preictal, interictal_hour, 0)

                    interictal_data = retrive_data(recordings, in_path, start_interictal, end_interictal)
                    preictal_data = retrive_data(recordings, in_path, start_preictal, end_preictal)

                    if segment_size > 0:
                        interictal_data = segment_data(interictal_data, segment_size, rec.sampling_rate)
                        preictal_data = segment_data(preictal_data, segment_size, rec.sampling_rate)

                    data.append((interictal_data, preictal_data))
                
                last_seizure_end = s[1]

    if len(data) < 1:
        print(f'No dataset created for {patient.id}\n')
        return None

    if balance:
        data = balance_data(data)
    
    if split:
        train_data, train_labels, test_data, test_labels = split_data(data)

        data = {'train_data': train_data, 
                'train_labels': train_labels, 
                'test_data': test_data, 
                'test_labels': test_labels, 
                'channels': np.array(recordings[0].channels)}
    
        print(f'Training data shape: {data['train_data'].shape}')
        print(f'Test data shape: {data['test_data'].shape}')
        print(f'Gap size: {gap_size}')
    else:
        labels = [np.append(np.zeros(d[0].shape[0]), np.ones(d[1].shape[0])) for d in data]
        data = {'data': np.vstack(np.array([np.vstack((d[0], d[1])) for d in data])),
                'labels': np.hstack((labels)),
                'channels': np.array(recordings[0].channels)}
        
        print(f'Data shape: {data['data'].shape}')
    
    if save:
        start_time = time.time()
        print(f'Creating file for {patient.id}...')

        filename = f'{out_path}/{patient.id}.npz'
        np.savez(filename, **data)
        
        print(f'File created in {time.time() - start_time:.2f} seconds\n')

    return data


def split_data(data, train_size = 80, test_size = 20):
    """
    Splits the given data into training and testing sets based on the specified train size and test size.
    
    Args:
        data: the input data to be split.
        train_size (optional): the percentage of data to be used for training. Default is 80.
        test_size (optional): the percentage of data to be used for testing. Default is 20.

    Returns:
        tuple: a tuple containing:
            train_data: the training data.
            train_labels: the labels corresponding to the training data.
            test_data: the testing data.
            test_labels: the labels corresponding to the testing data.
    """

    if train_size + test_size > 100: return

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    rng = np.random.default_rng()
    for d in data:
        n_training = ((len(d[0]) + len(d[1])) * (train_size)) // 100

        interictal_train_indices, interictal_test_indices = split_indices_randomly(np.arange(len(d[0])), n_training // 2, rng)
        preictal_train_indices, preictal_test_indices = split_indices_randomly(np.arange(len(d[1])), n_training // 2, rng)

        train_data.append(
            np.vstack(
                (np.array([d[0][i] for i in interictal_train_indices]), 
                 np.array([d[1][i] for i in preictal_train_indices]))
            )
        )
        
        train_labels.append(
            np.append(
                np.zeros(len(interictal_train_indices)),
                np.ones(len(preictal_train_indices))
            )
        )

        test_data.append(
            np.vstack(
                (np.array([d[0][i] for i in interictal_test_indices]), 
                 np.array([d[1][i] for i in preictal_test_indices]))
            )
        )

        test_labels.append(
            np.append(
                np.zeros(len(interictal_test_indices)),
                np.ones(len(preictal_test_indices))
            )
        )

    return (np.vstack((train_data)), np.hstack((train_labels)), np.vstack((test_data)), np.hstack((test_labels)))


def split_indices_randomly(array, size, rng):
    """
    Split the array into two subarrays, the first one with the specified size and the second one with the remaining elements.

    Args:
        array (np.array): array to split.
        size (int): size of the first subarray.
        rng (np.random.Generator): random number generator.

    Returns:
        tuple: tuple with the first subarray and the second subarray.
    """

    first_sub = rng.choice(array, size, replace = False)
    second_sub = np.setdiff1d(array, first_sub)

    return first_sub, second_sub


def balance_data(data):
    """
    Balance the data with the same number of interictal and preictal segments.

    Args:
        data (list): list of tuples with the interictal and preictal data.

    Returns:
        list: list of tuples with the balanced data.
    """

    # balanced_data = []
    # for d in data:
    #     interictal_index = np.arange((len(d[1])))
    #     interictal = np.array([d[0][i] for i in interictal_index])

    #     balanced_data.append((interictal, d[1]))

    balanced_data = []
    rng = np.random.default_rng()
    for d in data:
        random_interictal_index = rng.choice(len(d[0]), size = len(d[1]), replace = False)
        random_interictal = np.array([d[0][i] for i in random_interictal_index])

        balanced_data.append((random_interictal, d[1]))

    return balanced_data


def segment_data(data, segment_size, sampling_rate):
    """
    Segment the data in segments of the specified size. 

    Args:
        data (np.array): data to segment.
        segment_size_seconds (int): size of the segments in seconds.
        sampling_rate (int): sampling rate of the data.

    Returns:
        np.array: segmented data.
    """
    n_samples = sampling_rate * segment_size
    n_segments = len(data) // n_samples

    return np.array([data[i * n_samples:(i + 1) * n_samples] for i in range(n_segments)])


def retrive_data(recordings, in_path, start_datetime, end_datetime):
    """
    Retrieve the data of the patient between the specified datetimes.

    Args:
        recordings (list): list of EEGrec objects.
        in_path (str): path where the recordings are stored.
        start_datetime (datetime): start datetime.
        end_datetime (datetime): end datetime.

    Returns:
        np.array: data of the patient between the specified datetimes.
    """

    for i, rec in enumerate(recordings):
        if rec.start <= start_datetime <= rec.end:
            if rec.start <= end_datetime <= rec.end:
                return rec.retrive_data(f'{in_path}/{rec.id}.edf', 
                                        start_seconds = (start_datetime - rec.start).total_seconds(), 
                                        end_seconds = (end_datetime - rec.start).total_seconds())
            else:             
                data = rec.retrive_data(f'{in_path}/{rec.id}.edf', start_seconds = (start_datetime - rec.start).total_seconds())
                break
        
    for rec in recordings[i+1:]:
        if rec.start <= end_datetime <= rec.end:
            rec_data = rec.retrive_data(f'{in_path}/{rec.id}.edf', end_seconds = (end_datetime - rec.start).total_seconds())
            data = np.vstack((data, rec_data))
            return data
        else:
            rec_data = rec.retrive_data(f'{in_path}/{rec.id}.edf')
            data = np.vstack((data, rec_data))


def check_datetime(recordings, phase_datetime, control = 'start'):
    """
    Check if the phase datetime is within the recordings. If not, return the closest datetime.

    Args:
        recordings (list): list of EEGrec objects.
        phase_datetime (datetime): datetime to check.
        control (str, optional): control to return the closest datetime. Defaults to 'start'.

    Returns:
        datetime: closest datetime to the phase datetime.
    """
    min_diff = float('inf')
    min_index = None

    for i, rec in enumerate(recordings):
        if rec.start <= phase_datetime <= rec.end:
            return phase_datetime

        diff = abs(rec.start - phase_datetime).total_seconds()
        if diff < min_diff:
            min_diff = diff
            min_index = i

    if control.lower() == 'end' and min_index > 0:
        return recordings[min_index - 1].end

    return recordings[min_index].start


def get_phase_datetimes(recordings, reference_datetime, duration, gap):
    """
    Returns the start and end datetimes for a given phase of recordings.
    
    Args:
        recordings (list): list of recording datetimes.
        reference_datetime (datetime): reference datetime for the phase.
        duration (int): duration of the phase in hours.
        gap (int): gap in seconds before the end datetime.

    Returns:
        tuple: a tuple containing the start and end datetimes for the phase.
    """

    end = reference_datetime - timedelta(seconds = gap)
    start = end - timedelta(hours = duration)

    return (check_datetime(recordings, start, control = 'start'), check_datetime(recordings, end, control = 'end'))


def max_channel_recordings(patient):
    """
    Filters and retains the group of recordings with the same channels that have the maximum number of preictal and interictal hours.

    Args:
        patient (Patient): patient object.

    Returns:
        tuple: tuple with the recordings and the gap in hours between seizures that maximizes the number of preictal and interictal hours.
    """

    recordings = []
    grouped_recordings = patient.group_by_channels()
    for recs_indexes in grouped_recordings.values():
        #recs = patient.recordings[recs_indexes[0]:recs_indexes[-1] + 1] #TODO
        recs = [patient.recordings[i] for i in recs_indexes]
        consecutive_recs_group = get_consecutive_recordings(recs, range_minutes=get_max_gap_within_threshold(recs))
        gaps = []
        for cs in consecutive_recs_group:
            gaps.extend(seizure_gaps(cs))

        mean_gap = min(np.mean(gaps) if len(gaps) > 0 else 0, 5)

        h1 = count_greater(gaps, mean_gap) * mean_gap
        h2 = count_greater(gaps, 5) * 5
        if h1 > h2:
            recordings.append((consecutive_recs_group, mean_gap, h1))
        else:
            recordings.append((consecutive_recs_group, 5, h2))

            
    return max(recordings, key=lambda x: x[-1])[:-1]


def count_greater(values, threshold):
    """
    Count the number of values greater than or equal to the threshold.

    Args:
        values (list): list of values.
        threshold (int): threshold to compare the values with.

    Returns:
        int: number of values greater than or equal to the threshold.
    """
    return sum(1 for v in values if v >= threshold)


def get_consecutive_recordings(recordings, range_minutes = 5):
    """
    Get the consecutive recordings that are separated by a gap greater than the specified range.

    Args:
        recordings (list): list of EEGrec objects.
        range_minutes (int, optional): range in minutes to consider two recordings as consecutive. Defaults to 5.

    Returns:
        list: list of lists with the consecutive recordings.
    """
    if len(recordings) < 2: return [recordings]
    
    consecutive_recordings = []
    start_segment = 0
    for i, rec in enumerate(recordings[1:], 1):
        if rec.start - recordings[i - 1].end > timedelta(minutes = range_minutes):
            consecutive_recordings.append(recordings[start_segment:i])
            start_segment = i

    if start_segment == i:
        consecutive_recordings.append([recordings[-1]])
    else:
        consecutive_recordings.append(recordings[start_segment:])

    return consecutive_recordings


def get_max_gap_within_threshold(recordings, max_diff_minutes = 10):
    """
    Get the maximum gap between two recordings that is less than the specified threshold.

    Args:
        recordings (list): list of EEGrec objects.
        max_diff_minutes (int, optional): maximum difference in minutes between two recordings to consider them continuous. Defaults to 10.

    Returns:
        int: maximum gap between two recordings in minutes that is less than the specified threshold.
    """

    if len(recordings) < 2: return 0

    diff = [(recordings[i + 1].start - rec.end).total_seconds() / 60 for i, rec in enumerate(recordings[:-1])]
    filtered_diff = tuple(filter(lambda x: x <= max_diff_minutes, diff))

    return max(filtered_diff)


def seizure_gaps(recordings):
    """
    Calculates the time gaps in hours between seizures.

    Args:
        recordings (list): list of EEGrec objects.

    Returns:
        list: list of time gaps in hours between seizures.
    """

    gaps = []
    last_seizure_end = recordings[0].start
    for rec in recordings:
        for s in rec.get_seizures_datetimes():
            gaps.append((s[0] - last_seizure_end).total_seconds() / 3600)
            last_seizure_end = s[1]

    return gaps