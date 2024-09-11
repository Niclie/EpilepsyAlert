from scripts import run_preprocessing, run_prediction, run_training
from src.utils import constants, debug_utils
from src.data_preprocessing import load_data
from src.visualization import visualize
import keras
import numpy as np
import src.model.train_model as tm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def main():
    # patient_id = 'chb01'
    # # run(patient_id, model='mlp')
    # m = load_and_valuate(patient_id, 'cnn')
    
    # f = open(f'{constants.RESULTS_FOLDER}/results.csv', 'a')
    # f.write(f'{round(m[0], 2)}, {round(m[1], 2)}, {round(m[2], 2)}\n')
    # f.close()
    
    
    patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19','chb20', 'chb21', 'chb22', 'chb23']
    f = open(f'{constants.RESULTS_FOLDER}/results.csv', 'a')
    # tot_acc = []
    # tot_loss = []
    for patient_id in patients:
        m = load_and_valuate(patient_id, 'cnn')
        if m is None: continue
        f.write(f'{round(m[0], 2)}, {round(m[1], 2)}, {round(m[2], 2)}\n')
    f.close()
        
    #     val = run(patient_id, 'lstm', model_path = f'{constants.MODELS_FOLDER}/test', plot_path = f'{constants.PLOTS_FOLDER}/test')
    #     if val is None: continue
    #     tot_acc.append(val['accuracy'])
    #     tot_loss.append(val['loss'])
    
    # print(f'Mean accuracy: {np.mean(tot_acc)}')
    # print(f'Mean loss: {np.mean(tot_loss)}')
    

def load_and_valuate(patient_id, model_type,  model_path = constants.MODELS_FOLDER):
    dataset = run_preprocessing.get_dataset(patient_id)
    if dataset is None: return None
    model = keras.models.load_model(f'{model_path}/{model_type}/{patient_id}.keras')
    
    y_pred_prob = model.predict(dataset['test_data'])
    y_pred = (y_pred_prob > 0.5).astype(int)
    print(classification_report(dataset['test_labels'], y_pred, target_names=['Interictal', 'Preictal']))
    
    # Precision
    precision = precision_score(dataset['test_labels'], y_pred)

    # Recall
    recall = recall_score(dataset['test_labels'], y_pred)

    # F1-Score
    f1 = f1_score(dataset['test_labels'], y_pred)

    return precision, recall, f1
    

def run(patient_id, model, load_from_file = True, patience = 50, model_path = constants.MODELS_FOLDER, plot_path = constants.PLOTS_FOLDER):
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
    if dataset is None: return None

    match model:
        case 'mlp':
            history = run_training.run_training_mlp(dataset['train_data'], dataset['train_labels'], f'{model_path}/mlp', patient_id, patience = patience)
            model_type = 'mlp'
        
        case 'cnn':
            history = run_training.run_training_cnn(dataset['train_data'], dataset['train_labels'], f'{model_path}/cnn', patient_id, patience = patience)
            model_type = 'cnn'
            
        case 'lstm':
            history = run_training.run_training_lstm(dataset['train_data'], dataset['train_labels'], f'{model_path}/lstm', patient_id, patience = patience)
            model_type = 'lstm'
            
    visualize.plot_all_metrics(history, f'{plot_path}/{model_type}/{patient_id}', patient_id)

    model = keras.models.load_model(f'{model_path}/{model_type}/{patient_id}.keras')

    valuation = run_prediction.run_evaluation(model, dataset['test_data'], dataset['test_labels'])
    print(f'Accuracy: {valuation['accuracy']}\nLoss: {valuation['loss']}')
    visualize.log_metrics(patient_id, model_type, dataset['train_data'].shape[0], dataset['test_data'].shape[0], round(float(valuation['accuracy']), 2), round(float(valuation['loss']), 2))

    return valuation
    

if __name__ == '__main__':
    main()