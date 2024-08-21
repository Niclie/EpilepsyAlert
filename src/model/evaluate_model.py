def evaluate_model(model, test_data, test_label):
    """
    Evaluate the model with the test data and test label.

    Args:
        model (keras.Model): the model to evaluate.
        test_data (numpy.ndarray): the test data.
        test_label (numpy.ndarray): the test label.

    Returns:
        list: results of the evaluation.
    """

    evaluation = model.evaluate(test_data, test_label, return_dict=True)

    # results = []
    # for k in evaluation.keys():
    #     s = f'{k}: {evaluation[k]}'
    #     results.append(s)
    #     print(s)
        
    # return results
    return evaluation