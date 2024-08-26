import sys
import os
sys.path.append(os.path.abspath('.'))
from src.model.train_model import get_uncompiled_cnn, train_cnn


def run_training_cnn(training_data, label, out_path, patience = 50):
    """
    Train a model with the given data and label.

    Args:
        training_data (numpy.ndarray): the input training data.
        label (numpy.ndarray): the corresponding labels for the training data.
        out_path (str): the path to save the trained model.

    Returns:
        dict: a dictionary containing the training history.
    """

    model = get_uncompiled_cnn(input_shape=training_data.shape[1:])
    history = train_cnn(model, training_data, label, out_path, early_stopping_patience = patience)

    return history

# def run_training_cnn_rnn(training_data, label, out_path):
#     """
#     Train a model with the given data and label.

#     Args:
#         training_data (numpy.ndarray): the input training data.
#         label (numpy.ndarray): the corresponding labels for the training data.
#         out_path (str): the path to save the trained model.

#     Returns:
#         dict: a dictionary containing the training history.
#     """

#     model = get_uncompiled_cnn_rnn_model(input_shape=training_data.shape[1:])
#     history = train_cnn(model, training_data, label, out_path)

#     return history