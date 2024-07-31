import matplotlib.pyplot as plt


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