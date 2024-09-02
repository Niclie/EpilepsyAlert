import matplotlib.pyplot as plt
from src.utils import constants
import os.path
from datetime import date, datetime


def log_metrics(patient_id, n_training, n_test, accuracy, loss, file_path = f'{constants.RESULTS_FOLDER}/results.csv'):
    """
    Log the metrics of the model in a csv file.

    Args:
        patient_id (str): patient id.
        n_training (int): number of training examples.
        n_test (int): number of test examples.
        accuracy (float): accuracy of the model.
        loss (float): loss of the model.
        file_path (str, optional): path to the csv file. Defaults to f'{constants.RESULTS_FOLDER}/results.csv'.
    """
    if not os.path.isfile(file_path):
        f = open(file_path, 'w')
        f.write('ID, Training examples, Test examples, Accuracy, Loss, Date, Time\n')
    else:
        f = open(file_path, 'a')
        
    day = date.today().strftime('%d/%m/%Y')
    time = datetime.now().strftime('%H:%M:%S')
    f.write(f'{patient_id}, {n_training}, {n_test}, {accuracy}, {loss}, {day}, {time}\n')

    f.close()


def visualize_data(data, label, classes):
    """
    Visualizes the data for each class.

    Args:
        data (numpy.ndarray): input data.
        label (numpy.ndarray): labels corresponding to the data.
        classes (list): list of classes.
    """

    plt.figure()
    for c in classes:
        class_data = data[label == c]
        plt.plot(class_data[0].T[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()


def plot_all_metrics(history, file_name):
    """
    Plot all metrics from the given history object and save the plots with the specified file name.

    Args:
        history (dictionary): history object containing the training metrics.
        file_name (str): name of the file to save the plots.
    """

    keys = list(history.history.keys())
    keys = keys[:len(keys)//2]
    for metric in keys:
        plot_metric(history, metric, f'{file_name}_{metric}')


def plot_metric(history, metric, file_name):
    """
    Plot the specified metric from the given history object and save the plot with the specified file name.

    Args:
        history (dictionary): history object containing the training metrics.
        metric (str): metric to be plotted.
        file_name (str): name of the file to save the plot.
    """

    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    
    plt.savefig(file_name)
    plt.close()