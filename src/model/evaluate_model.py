from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def evaluate_model(actuals, predicted_classes, predicted_proba):
    """
    Returns the following metrics given the actuals, predicted classes and predicted probabilities:
    - Precision
    - Recall
    - F1 Score
    - auc

    Args:
        actuals (np.array): array of actual values of the target variable
        predicted_classes (np.array): array of predicted classes
        predicted_proba (np.array): array of predicted probabilities

    Returns:
        dict: dictionary containing the evaluation metrics
    """
    precision = precision_score(actuals, predicted_classes)
    recall = recall_score(actuals, predicted_classes) 
    f1 = f1_score(actuals, predicted_classes)    
    auc = roc_auc_score(actuals, predicted_proba)
    accuracy = accuracy_score(actuals, predicted_classes)

    return {
        'precision': f'{precision:.2f}',
        'recall': f'{recall:.2f}',
        'f1': f'{f1:.2f}',
        'auc': f'{auc:.2f}',
        'accuracy': f'{accuracy:.2f}'
    }