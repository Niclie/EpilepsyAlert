import sys
import os
sys.path.append(os.path.abspath('.'))
from src.model.predict import predict
from src.model.evaluate_model import evaluate_model


def run_evaluation(model, data, labels):
    """
    Run evaluation on the given model using the provided data and labels.

    Args:
        model (keras.Model): the model to be evaluated.
        data (numpy.ndarray): the input data for evaluation.
        labels (numpy.ndarray): the corresponding labels for the input data.
    """

    e = evaluate_model(model, data, labels) 
    return e


def run_prediction(model, data):
    """
    Run prediction using the given model on the provided data.

    Args:
        model (keras.Model): the trained model object to use for prediction.
        data (numpy.ndarray): the input data to make predictions on.
    """
    
    predict(model, data)
