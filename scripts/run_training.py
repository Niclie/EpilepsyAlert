from src.model.train_model import cnn


def run_training_cnn(training_data, label, out_path, file_name):
    """
    Script to run training for CNN

    Args:
        training_data (np.array): data to train the model
        label (np.array): label for the training data
        out_path (str): path to save the model
        file_name (str): name of the model file

    Returns:
        history: training history
    """
    return cnn(training_data, label, out_path, file_name)