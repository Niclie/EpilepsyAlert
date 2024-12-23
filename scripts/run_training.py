from src.model.train_model import load_model, train_model


def run_training_cnn(training_data, training_label, out_path, batch_size=64, epochs=500, early_stopping=50):
    """
    Script to run training for CNN

    Args:
        training_data (np.array): data to train the model
        training_label (np.array): label for the training data
        out_path (str): path to save the model
        batch_size(int): batch size for training. Default is 64.
        epochs (int): number of epochs for training. Default is 500.
        early_stopping (int): number of epochs for early stopping. Default is 50.
    Returns:
        history: training history
    """
    model = load_model(training_data[0].shape)
    history = train_model(model, training_data, training_label, out_path, batch_size, epochs, early_stopping)

    return history