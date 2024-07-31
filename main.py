from scripts import run_preprocessing, run_prediction, run_training
import src.utils.constants as constants
import src.visualization.visualize as visualize
import keras


def run(patient_id):
    dataset = run_preprocessing.get_dataset(patient_id, load_from_file=True) #, load_from_file=True
    training_dataset = dataset['training_data']
    training_label = dataset['training_label']
    test_dataset = dataset['test_data']
    test_label = dataset['test_label']

    #CNN
    history = run_training.run_training_cnn(training_dataset, training_label, f'{constants.MODELS_FOLDER}/{patient_id}_cnn')
    visualize.plot_all_metrics(history, f'{constants.PLOTS_FOLDER}/{patient_id}/{patient_id}_cnn')

    model = keras.models.load_model(f'{constants.MODELS_FOLDER}/{patient_id}_cnn.keras')
    
    f = open("output.txt", "a")
    f.write(patient_id + '_cnn '  + str(run_prediction.run_evaluation(model, test_dataset, test_label)) + "\n")
    f.close()


    #CNN_RNN
    history = run_training.run_training_cnn_rnn(training_dataset, training_label, f'{constants.MODELS_FOLDER}/{patient_id}_cnn_rnn')
    visualize.plot_all_metrics(history, f'{constants.PLOTS_FOLDER}/{patient_id}/{patient_id}_cnn_rnn')

    model = keras.models.load_model(f'{constants.MODELS_FOLDER}/{patient_id}_cnn_rnn.keras')
    
    f = open("output.txt", "a")
    f.write(patient_id + '_cnn_rnn ' + str(run_prediction.run_evaluation(model, test_dataset, test_label)) + "\n")
    f.close()


def main():
    patients = ['chb01', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb15', 'chb16', 'chb18', 'chb21', 'chb22']
    for patient_id in patients:
        run(patient_id)

if __name__ == '__main__':
    main()