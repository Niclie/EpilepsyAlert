import os
import matplotlib.pyplot as plt
from src.utils import constants
from datetime import date, datetime
from src.utils.debug_utils import check_folder


def log_metrics(patient_id, model, n_training, n_test, metrics, file_path=f'{constants.RESULTS_FOLDER}/results.csv'):
    """
    Log the metrics of the model in a csv file.

    Args:
        patient_id (str): patient id.
        model (str): model name.
        n_training (int): number of training examples.
        n_test (int): number of test examples.
        metrics (dictionary): dictionary containing the metrics of the model.
        file_path (str, optional): path to the csv file. Defaults to f'{constants.RESULTS_FOLDER}/results.csv'.
    """
    if not os.path.isfile(file_path):
        f = open(file_path, 'w')
        m = ', '.join(metrics.keys())
        f.write(f'ID, Model, Training examples, Test examples, {m}, Date, Time\n')
    else:
        f = open(file_path, 'a')
        
    day = date.today().strftime('%d/%m/%Y')
    time = datetime.now().strftime('%H:%M:%S')
    v = ', '.join(map(str, metrics.values()))
    f.write(f'{patient_id}, {model}, {n_training}, {n_test}, {v}, {day}, {time}\n')

    f.close()


def plot_all_metrics(history, best_epoch, out_path, file_name):
    """
    Plot all metrics from the given history object and save the plots with the specified file name.

    Args:
        history (dictionary): history object containing the training metrics.
        best_epoch (int): epoch with the best validation loss.
        out_path: (str): path to save the plots.
        file_name (str): name of the file to save the plots.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    keys = list(history.keys())
    keys = keys[:len(keys)//2]
    for metric in keys:
        __tf_plot_metric(history, best_epoch, metric, out_path, f'{file_name}_{metric}')


def __tf_plot_metric(history, best_epoch, metric, path, file_name):
    """
    Plot the specified metric from the given history object and save the plot with the specified file name.

    Args:
        history (dictionary): history object containing the training metrics.
        metric (str): metric to be plotted.
        path (str): path to save the plot.
        file_name (str): name of the file to save the plot.
    """
    check_folder(path)
    
    plt.figure()
    plt.plot(history[metric])
    plt.plot(history["val_" + metric])
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Saved Model (Epoch {best_epoch})')
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    
    plt.savefig(f'{path}/{file_name}')
    plt.close()