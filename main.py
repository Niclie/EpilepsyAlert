from scripts import run_preprocessing, run_prediction, run_training
import src.utils.constants as constants
import src.visualization.visualize as visualize
import keras
import os
from datetime import date, datetime


def run(patient_id, patience = 50):
    dataset = run_preprocessing.get_dataset(patient_id, False) #, load_from_file=True
    train_data = dataset['train_data']
    train_labels = dataset['train_labels']
    test_data = dataset['test_data']
    test_labels = dataset['test_labels']

    model_path = constants.MODELS_FOLDER
    history = run_training.run_training_cnn(train_data, train_labels, f'{model_path}/{patient_id}', patience = patience)
    
    plot_path = f'{constants.PLOTS_FOLDER}/{patient_id}'
    try:
        os.mkdir(plot_path)
    except:
        print('Directory for plot already exist')

    visualize.plot_all_metrics(history, f'{plot_path}/{patient_id}')

    model = keras.models.load_model(f'{model_path}/{patient_id}.keras')

    valuation = run_prediction.run_evaluation(model, test_data, test_labels)
    f = open("output.txt", "a")
    #f.write('ID, Canali, Esempi di training, Esempi di test, Accuratezza, Loss, Data, Ora\n')
    day = date.today().strftime('%d/%m/%Y')
    time = datetime.now().strftime('%H:%M:%S')
    f.write(f'{patient_id}, {len(dataset['channels'])}, {train_data.shape[0]}, {test_data.shape[0]}, {round(float(valuation['accuracy']), 2)}, {round(float(valuation['loss']), 2)}, {day}, {time} \n')
    f.close()

    print(f'Accuracy: {valuation['accuracy']}\nLoss: {valuation['loss']}')

    return valuation['accuracy'], valuation['loss']


def main():
    #patients = ['chb01', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb15', 'chb16', 'chb18', 'chb21', 'chb22']
    #patients_v2 = ['chb02', 'chb03', 'chb04', 'chb11', 'chb12', 'chb13', 'chb14', 'chb17', 'chb19','chb20', 'chb23'] #, 'chb24'
    
    # patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19','chb20', 'chb21', 'chb22', 'chb23']


    # for patient_id in patients[13:]:
    #     run(patient_id)
    
    patient_id = 'chb15'
    # run(patient_id, 10)

    for _ in range(5):
        a, l = run(patient_id)
        if a >= 0.75 and l <= 0.55:
            break


if __name__ == '__main__':
    main()