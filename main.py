from src.utils.constants import MODELS_FOLDER, PLOTS_FOLDER, PATIENTS
from scripts.run_training import run_training_cnn
from scripts.run_preprocessing import run_preprocessing
from scripts.run_prediction import run_evaluation
from src.visualization.visualize import plot_all_metrics, log_metrics
from keras.api.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    #chb01, chb02, chb03, chb04, chb06
    patient_id = 'chb01'
    try:
        run(patient_id)
    except FileNotFoundError as err:
        print(err)
    # for patient_id in PATIENTS[]:
    #     try:
    #         run(patient_id)
    #     except AttributeError:
    #         continue


def run(patient_id, load_from_file=True):
    """
    Run the whole pipeline for a given patient_id.

    Args:
        patient_id (str): patient identifier.
        load_from_file (bool, optional): whether to load the dataset from a file. Defaults to True.
    """
    dataset = run_preprocessing(patient_id, load_from_file)

    history = run_training_cnn(dataset['train_data'], dataset['train_labels'], f'{MODELS_FOLDER}/cnn', patient_id)
    best_epoch = np.argmin(history.history['val_loss'])
    plot_all_metrics(history, best_epoch, f'{PLOTS_FOLDER}/cnn/{patient_id}', f'{patient_id}_cnn')
    
    model = load_model(f'{MODELS_FOLDER}/cnn/{patient_id}.keras')
    predicted_proba = model.predict(dataset['test_data'])
    predicted_classes = (predicted_proba > 0.5).astype(int)
    ev = run_evaluation(dataset['test_labels'], predicted_classes, predicted_proba)
    log_metrics(patient_id, 'CNN', len(dataset['train_data']), len(dataset['test_data']), ev)
    print(ev)


if __name__ == '__main__':
    main()
