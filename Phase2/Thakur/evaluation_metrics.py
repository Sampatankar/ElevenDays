import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

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

# actual targets:
y_true = [
    0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1
]

# predicted probabilities of a sample being 1:
y_pred = [
    0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99
]

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



