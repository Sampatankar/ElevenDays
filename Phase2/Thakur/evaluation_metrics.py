"""
This is the initial file relating to the Evaluation Metrics section from pg.30 in Thakur.


"""

# Accuracy:

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score

    """

    # initialise a simple counter for correct predictions
    correct_counter = 0

    # loop over all elements of y_true and y_pred together:
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            # if predicition is equal to truth, increase the counter
            correct_counter += 1

    # return accuracy which is correct predictions over the number of samples:
    return correct_counter / len(y_true)


# the sklearn version of accuracy:
from sklearn import metrics

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

# uncomment the print statement for this basic accuracy score:
# print(metrics.accuracy_score(l1,l2))


# TP, TN, FP, FP in code:

def true_positive(y_true, y_pred):
    """
    Function for true_positive
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: number of true positive

    """

    # initialise:
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1

    return tp

def true_negative(y_true, y_pred):
    """
    Function for true_negative
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: number of true negative

    """

    # initialise:
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    """
    Function for false_positive
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: number of false positive

    """

    # initialise:
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    """
    Function for false_negative
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: number of false negative

    """

    # initialise:
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

print(true_positive(l1, l2))
print(false_positive(l1, l2))
print(false_positive(l1, l2))
print(false_negative(l1, l2))



