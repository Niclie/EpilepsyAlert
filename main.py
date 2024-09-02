from scripts import run_preprocessing, run_prediction, run_training
from src.utils import constants
from src.visualization import visualize
import keras


def main():
    patient_id = 'chb01'
    run(patient_id)
    
    #patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19','chb20', 'chb21', 'chb22', 'chb23']

    # for patient_id in patients:
    #     run(patient_id)
    

def run(patient_id, load_from_file = True, patience = 50, model_path = constants.MODELS_FOLDER, plot_path = constants.PLOTS_FOLDER):
    """
    Run the whole pipeline for a given patient_id.

    Args:
        patient_id (str): patient id.
        load_from_file (bool, optional): flag to load the dataset from file. Defaults to True.
        patience (int, optional): patience for early stopping. Defaults to 50.
        model_path (str, optional): path to save the model. Defaults to constants.MODELS_FOLDER.
        plot_path (str, optional): path to save the plots. Defaults to constants.PLOTS_FOLDER.
    """
    dataset = run_preprocessing.get_dataset(patient_id, load_from_file)
    if dataset is None: return

    history = run_training.run_training_cnn(dataset['train_data'], dataset['train_labels'], f'{model_path}/{patient_id}', patience = patience)
    
    visualize.plot_all_metrics(history, f'{plot_path}/{patient_id}/{patient_id}')

    model = keras.models.load_model(f'{model_path}/{patient_id}.keras')

    valuation = run_prediction.run_evaluation(model, dataset['test_data'], dataset['test_labels'])
    print(f'Accuracy: {valuation['accuracy']}\nLoss: {valuation['loss']}')
    visualize.log_metrics(patient_id, dataset['train_data'].shape[0], dataset['test_data'].shape[0], round(float(valuation['accuracy']), 2), round(float(valuation['loss'])))


if __name__ == '__main__':
    main()