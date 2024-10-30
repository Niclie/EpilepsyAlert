from scripts.run_preprocessing import get_dataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import tsfel
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier


def main():
    patient_id = 'chb01'
    dataset = get_dataset(patient_id)
    # x_train = dataset['train_data']
    # x_test = dataset['test_data']
    y_train = dataset['train_labels']
    y_test = dataset['test_labels']
    
    x_train = pd.read_csv(f'{patient_id}_x_train.csv')
    x_test = pd.read_csv(f'{patient_id}_x_test.csv')
    
    model = logistic_reg_train_evaluate(x_train, y_train, x_test, y_test)
    print_learning_curve(model, x_train, y_train)

    return



    
def logistic_reg_train_evaluate(x_train, y_train, x_test, y_test):
    #model = DecisionTreeClassifier(max_depth=3)
    model = LogisticRegression()
    #model = RandomForestClassifier(n_estimators=50) anche no
    #model = GaussianNB() anche no
    #model = GradientBoostingClassifier(n_estimators=100) ensomma
    #model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200) no
    
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    p_naive = y_test.mean()
    y_naive_pred = np.full_like(y_test, p_naive)  # previsione naive (stessa probabilità per tutte le osservazioni)
    naive_log_loss = log_loss(y_test, y_naive_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f'Naive Log Loss: {naive_log_loss:.2f}')
    print(f'Log Loss: {logloss:.2f}')
    return model
    
    
def print_learning_curve(model, x_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_train, y_train, cv=10, scoring=log_loss_scorer, shuffle=True
    ) #neg_log_loss
    
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
    
    
def preprocess_data(data, fs, save=False, file_name = None):
    cfg = tsfel.get_features_by_domain(json_path='feature.json')
    x = tsfel.time_series_features_extractor(cfg, data, fs=fs)
    if save and file_name is not None: x.to_csv(f'{file_name}.csv', index=False)
    return x

def log_loss_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)
    return log_loss(y, y_pred_proba)