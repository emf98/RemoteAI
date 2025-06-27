##definition statements for statistics 
import sklearn
from sklearn.metrics import brier_score_loss
from sklearn.metrics import recall_score, precision_score
import xarray as xr
import numpy as np

####################################################
# functions to make RAS and PAS more concise
def get_class_preds(probs):
    #Convert softmax probs to one-hot class predictions for precision and recall.
    class_indices = np.argmax(probs, axis=1)
    return np.eye(2)[class_indices]

def compute_scores(iter_idx, metric_fn, score_matrix, true_labels, pred_labels, climo):
    #this is to append the iteration to the appropriate index for CV
    for j in range(climo.shape[1]):
        score = metric_fn(true_labels[:, j], pred_labels[:, j])
        score_matrix[iter_idx, j] = round(score, 4)

###################################################
# #Brier Skill Scores
def BSS(y_in,pred_in):
    #convert one-hot labels to 1D integer labels (0 or 1)
    y_true = np.argmax(y_in, axis=1)

    #predicted probability for the positive class (class 1)
    y_prob = pred_in[:,1]

    #climatology baseline: constant probability = mean positive class rate
    p_climatology = np.full_like(y_true, y_true.mean(), dtype=float)

    #Brier Scores
    bs_model = brier_score_loss(y_true, y_prob)
    bs_climo = brier_score_loss(y_true, p_climatology)

    #Brier Skill Score
    bss = 1 - (bs_model / bs_climo)         
    
    return bss;

###################################################
# #Recall Accuracy Scores
def RAS(iter_idx, Rec_all, climo_full, Y_all, pred,
            climo_val, Rec_val, Y_val, pred_val,
            climo_train, Rec_train, Y_train, pred_train,
            climo_test, Rec_test, Y_test, pred_test):

    compute_scores(iter_idx, recall_score, Rec_all, Y_all, get_class_preds(pred), climo_full)
    compute_scores(iter_idx, recall_score, Rec_val, Y_val, get_class_preds(pred_val), climo_val)
    compute_scores(iter_idx, recall_score, Rec_train, Y_train, get_class_preds(pred_train), climo_train)
    compute_scores(iter_idx, recall_score, Rec_test, Y_test, get_class_preds(pred_test), climo_test)

    return Rec_all, Rec_val, Rec_train, Rec_test

###################################################
# #Precision Accuracy Scores
def PAS(iter_idx, Prec_all, climo_full, Y_all, pred,
            climo_val, Prec_val, Y_val, pred_val,
            climo_train, Prec_train, Y_train, pred_train,
            climo_test, Prec_test, Y_test, pred_test):

    compute_scores(iter_idx, precision_score, Prec_all, Y_all, get_class_preds(pred), climo_full)
    compute_scores(iter_idx, precision_score, Prec_val, Y_val, get_class_preds(pred_val), climo_val)
    compute_scores(iter_idx, precision_score, Prec_train, Y_train, get_class_preds(pred_train), climo_train)
    compute_scores(iter_idx, precision_score, Prec_test, Y_test, get_class_preds(pred_test), climo_test)

    return Prec_all, Prec_val, Prec_train, Prec_test
