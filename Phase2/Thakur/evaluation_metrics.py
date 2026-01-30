import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
from collections import Counter

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


# uncomment the print statement for this basic accuracy score:
# print(true_positive(l1, l2))
# print(false_positive(l1, l2))
# print(false_positive(l1, l2))
# print(false_negative(l1, l2))


# Accuracy implementing TP, TN, FP and FN:

def accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy using tp/tn/fp/fn
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score

    """

    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score


# uncomment the print statement for this basic accuracy score:
# print(accuracy(l1, l2))
# print(accuracy_v2(l1, l2))
# # the sklearn metrics method:
# print(metrics.accuracy_score(l1, l2))



# Implementing Precision in code:
def precision(y_true, y_pred):
    """
    Function to calculate precison using tp and fp
    
    :param y_true: List of true values
    :param y_pred: List of predicted values
    :return: precision score

    """

    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp / (tp+ fp)
    return precision

# uncomment the print statement for this basic accuracy score:
# print(precision(l1, l2))


# Implementing Recall in code:
def recall(y_true, y_pred):
    """
    Function to calculate recall using tp and fn
    
    :param y_true: List of true values
    :param y_pred: List of predicted values
    :return: recall score

    """

    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

# uncomment the print statement for this basic accuracy score:
# print(recall(l1, l2))


# Implementing the Precision-Recall curve:
"""
y_true are our target values.  y_pred is the probability values for a sample being assigned a value of 1.  So, now, we look at probabilities in prediction instead of of the predicted value (which is most of the time calculated with a threshold at 0.5)
"""
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [
    0.02638412, 0.11114267, 0.31620708, 0.0490937, 0.0191491, 0.17554844, 0.15952202, 0.03819563, 0.11639273, 0.079377, 0.08584789, 0.39095342, 0.27259048, 0.03447096, 0.04644807, 0.03543574, 0.18521942, 0.05934905, 0.61977213, 0.33056815
]


precisions = []
recalls = []

thresholds = [
    0.0490937, 0.05934905, 0.079377, 0.08584789, 0.11114267, 0.11639273, 0.15952202, 0.17554844, 0.18521942, 0.27259048, 0.31620708, 0.33056815, 0.39095342, 0.61977213
]

# for every threshold, calculate predictions in binary and append calculated precisions and recalls to their respective lists:

for i in thresholds:
    temp_prediction = [1 if x >= i else 0 for x in y_pred]
    p = precision(y_true, temp_prediction)
    r = recall(y_true, temp_prediction)
    precisions.append(p)
    recalls.append(r)

# uncomment the print statement for this basic accuracy score:
# plt.figure(figsize=(7, 7))
# plt.plot(recalls, precisions)
# plt.xlabel('Recall', fontsize=15)
# plt.ylabel('Precision', fontsize=15)
# plt.show()



# Implement F1 score:
def f1(y_true, y_pred):
    """
    Function to implement f1 score:
    
    :param y_true: List of true values
    :param y_pred: List of predicted values
    :return: F1 score

    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    score = (2 * p * r) / (p + r)
    return score

# uncomment the print statement for this basic accuracy score:
# y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# print(f1(y_true, y_pred))

# # implement f1 from sklearn:
# print(metrics.f1_score(y_true, y_pred))



# Implementing recall as TPR/sensitivity:
def tpr(y_true, y_pred):
    """
    Function to implement recall as TPR
    
    :param y_true: List of true values
    :param y_pred: List of predicted values

    :return: TPR

    """

    return recall(y_true, y_pred)



# implementing FPR:
def fpr(y_true, y_pred):
    """
    Function to implement FPR
    
    :param y_true: List of true values
    :param y_pred: List of predicted values
    :return: FPR score

    """

    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp / (tn + fp)


# calculating TPR and FPR for varying thresholds
tpr_list = []
fpr_list = []

# actual targets:
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1:
y_pred = [
    0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99
]

# handmade thresholds:
thresholds = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0
]

# loop over all thresholds:
for threshold in thresholds:
    # calculate predictions for a given threshold:
    temp_pred = [1 if x >= threshold else 0 for x in y_pred]
    # calculate tpr:
    temp_tpr = tpr(y_true, temp_pred)
    # calculate fpr:
    temp_fpr = fpr(y_true, temp_pred)
    # append for tpr and fpr to lists
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)


# uncomment the print statement for this basic accuracy score:
# plt.figure(figsize=(7, 7))
# plt.fill_between(fpr_list, tpr_list, alpha=0.4)
# plt.plot(fpr_list, tpr_list, lw=3)
# plt.xlim(0, 1.0)
# plt.ylim(0, 1.0)
# plt.xlabel('FPR', fontsize=15)
# plt.ylabel('TPR', fontsize=15)
# plt.show()



# sklearn to calculate AUC from ROC plot:

# reusing y_true and y_pred arrays from above
# uncomment the print statement for this basic accuracy score:
# print(metrics.roc_auc_score(y_true, y_pred))


# selecting threshold from ROC plot:
# empty lists to store true positive and false positive values:

tp_list = []
fp_list = []

# actual targets: UNCOMMENT TO RUN THE FUNCTIONS BELOW UNTIL PRECISION & RECALL MULTICLASS
# y_true = [
#     0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1
# ]

# # predicted probabilities of a sample being 1:
# y_pred = [
#     0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99
# ]

# some handmade thresholds:
thresholds = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0
]

# loop over all thresholds:
for threshold in thresholds:
    # calculate predictions for a given threshold:
    temp_pred = [1 if x >= threshold else 0 for x in y_pred]
    # calculate tp:
    temp_tp = true_positive(y_true, temp_pred)
    # calculate fp:
    temp_fp = false_positive(y_true, temp_pred)
    # append to lists:
    tp_list.append(temp_tp)
    fp_list.append(temp_fp)


# Implementing a Log Loss function:
def log_loss(y_true, y_proba):
    """
    Function to calculate log_loss
    
    :param y_true: List of true values
    :param y_proba: List of probabilities for 1 
    :return: overall log loss

    """

    # define an epsilon value, which can also be used as input.  The value is used to clip probabilities:
    epsilon = 1e-15

    # initialise an empty list to store individual losses
    loss = []

    # loop over all true and predicted probability values:
    for yt, yp in zip(y_true, y_proba):
        # adjust probability, 0 gets converted to 1e-15, 1 gets converted to 1-1e15
        yp = np.clip(yp, epsilon, 1 - epsilon)
        # calculate loss for one sample
        temp_loss = - 1.0 * (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        # add to the loss list:
        loss.append(temp_loss)
    # return mean loss over all samples
    return np.mean(loss)

y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]


# uncomment the print statement for this basic accuracy score:
# print(log_loss(y_true, y_proba))

# # the SKlearn function:
# print(metrics.log_loss(y_true, y_proba))


# Implementation of Macro Averaged Precision:

def macro_precisiom(y_true, y_pred):
    """
    Function to calculate macro averaged precision:
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: macro precision score

    """

    # find the number of classes by taking the length of the number of unique values in the true list
    num_classes = len(np.unique(y_true))

    # initialise precision to 0:
    precision = 0

    # loop over all classes:
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class:
        tp = true_positive(temp_true, temp_pred)

        # calculate false positive for current class:
        fp = false_positive(temp_true, temp_pred)

        # calculate precision for current class:
        temp_precision = tp / (tp + fp)

        # keep adding precision for all classes:
        precision += temp_precision

    # calculate and return average precision over all classes:
    precision /= num_classes
    return precision



# Implementation of micro-averaged precision:

def micro_precision(y_true, y_pred):
    """
    Function to calculate micro-averaged precision
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: micro-averaged precision

    """

    # find the number of classes by taking the length of the number of unique values in the true list
    num_classes = len(np.unique(y_true))

    # initialise tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes:
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class and update overall tp
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision:
    precision = tp / (tp + fp)
    return precision


# Implementation of weighted precision:

def weighted_precision(y_true, y_pred):
    """
    Function to calculate Weighted Precision
    
    :param y_true: List of True Values
    :param y_pred: List of Predicted Values
    :return: weighted precision

    """

    
    # find the number of classes by taking the length of unique values in the true list
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary - output similar to: {0: 20, 1: 15, 2: 21}
    class_counts = Counter(y_true)

    # initialise precision to 0:
    precision = 0

    # loop over all classes:
    for class_ in range(num_classes):
        # all classes except the current are considered negative:
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate tp and fp for class:
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        # calculate precision of class:
        temp_precision = tp / (tp + fp)

        # multiple precision with count of samples in class:
        weighted_precision = class_counts[class_] * temp_precision

        # add to overall precision:
        precision += weighted_precision

    # calculate  overall precision by dividing by total number of samples:
    overall_precision = precision / len(y_true)

    return overall_precision


y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]

y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

# use of implementations and sklearn versions:
print(macro_precisiom(y_true, y_pred))
print(metrics.precision_score(y_true, y_pred, average="macro"))

print(micro_precision(y_true, y_pred))
print(metrics.precision_score(y_true, y_pred, average="micro"))

print(weighted_precision(y_true, y_pred))
print(metrics.precision_score(y_true, y_pred, average="weighted"))

