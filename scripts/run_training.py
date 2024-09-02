from src.model.train_model import build_cnn, build_lstm, build_resnet, train


def run_training_cnn(training_data, label, out_path, patience = 50):
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
    model = build_cnn(input_shape=training_data.shape[1:])

    return __train(model, training_data, label, out_path, early_stopping = patience)


def run_training_lstm(training_data, label, out_path, patience = 50):
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
    model = build_lstm(input_shape=training_data.shape[1:])

    return __train(model, training_data, label, out_path, early_stopping = patience)


def run_training_resnet(training_data, label, out_path, patience = 50):
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
    model = build_resnet(input_shape=training_data.shape[1:])

    return __train(model, training_data, label, out_path, early_stopping = patience)


def __train(model, training_data, label, out_path, early_stopping = 50):
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
    
    return train(model, training_data, label, out_path, early_stopping = early_stopping)