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

print(metrics.accuracy_score(l1,l2))

