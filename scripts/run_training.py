from src.utils.debug_utils import check_folder
from src.model.train_model import build_mlp, build_cnn, build_lstm, train

def run_training_mlp(training_data, label, out_path, file_name, patience = 50):
    check_folder(out_path)
    model = build_mlp(input_shape=training_data.shape[1:])

    return __train(model, training_data, label, out_path, file_name, optimizer = 'rmsprop' , early_stopping = patience)


def run_training_cnn(training_data, label, out_path, file_name, patience = 50):
    """
    Train a model with the given data and label.

    Args:
        training_data (numpy.ndarray): the input training data.
        label (numpy.ndarray): the corresponding labels for the training data.
        out_path (str): the path to save the trained model.
        patience (int, optional): patience for early stopping. Defaults to 50.

    Returns:
        dict: a dictionary containing the training history.
    """
    check_folder(out_path)
    model = build_cnn(input_shape=training_data.shape[1:])

    return __train(model, training_data, label, out_path, file_name, early_stopping = patience)


def run_training_lstm(training_data, label, out_path, file_name, patience = 50):
    """
    Train a model with the given data and label.

    Args:
        training_data (numpy.ndarray): the input training data.
        label (numpy.ndarray): the corresponding labels for the training data.
        out_path (str): the path to save the trained model.
        patience (int, optional): patience for early stopping. Defaults to 50.

    Returns:
        dict: a dictionary containing the training history.
    """
    check_folder(out_path)
    model = build_lstm(input_shape=training_data.shape[1:])

    return __train(model, training_data, label, out_path, file_name, early_stopping = patience, optimizer= 'rmsprop')


def __train(model, training_data, label, out_path, file_name, optimizer = 'adam', early_stopping = 50):
    """
    Train a model with the given data and label.

    Args:
        model (keras.Model): the model to train.
        training_data (numpy.ndarray): the input training data.
        label (numpy.ndarray): the corresponding labels for the training data.
        out_path (str): the path to save the trained model.
        early_stopping (int, optional): patience for early stopping. Defaults to 50.

    Returns:
        dict: a dictionary containing the training history.
    """
    
    return train(model, training_data, label, out_path, file_name, optimizer = optimizer, early_stopping = early_stopping)