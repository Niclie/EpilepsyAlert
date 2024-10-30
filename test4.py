from scripts.run_preprocessing import get_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


def rf_train_evaluate(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    return model


def print_learning_curve(model, x_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_train, y_train, cv=5, scoring=log_loss_scorer, shuffle=True
    )
    
    # Calcola la media e la deviazione standard per le prestazioni
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Traccia la curva per i punteggi di training
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Log-Loss")

    # Traccia la curva per i punteggi di validazione (cross-validation)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Log-Loss")

    plt.legend(loc="best")
    plt.show()
    
    
def log_loss_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)
    return log_loss(y, y_pred_proba)


def main():
    patient_id = 'chb01'
    dataset = get_dataset(patient_id)
    #x_train = dataset['train_data']
    y_train = dataset['train_labels']
    #x_test = dataset['test_data']
    y_test = dataset['test_labels']
    
    x_train = pd.read_csv(f'{patient_id}_x_train.csv')
    x_test = pd.read_csv(f'{patient_id}_x_test.csv')
    
    model = rf_train_evaluate(x_train, y_train, x_test, y_test)
    print_learning_curve(model, x_train, y_train)

    # x_train = pd.read_csv(f'{patient_id}_x_train.csv')
    # x_test = pd.read_csv(f'{patient_id}_x_test.csv')
    
    
if __name__ == '__main__':
    main()