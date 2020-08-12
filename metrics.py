import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc



def true_positives(y_true, y_pred_prob, th):
  TP = 0
  thresholded_probs = y_pred_prob >= th
  TP = np.sum((y_true==1) & (thresholded_probs==1))
  return TP



def true_negatives(y_true, y_pred_prob, th):
  TN = 0
  thresholded_probs = y_pred_prob >= th
  TN = np.sum((y_true==0) & (thresholded_probs==0))
  return TN



def false_positives(y_true, y_pred_prob, th):
  FP = 0
  thresholded_probs = y_pred_prob >= th
  FP = np.sum((y_true==0) & (thresholded_probs==1))
  return FP



def false_negatives(y_true, y_pred_prob, th):
  FN = 0
  thresholded_probs = y_pred_prob >= th
  FN = np.sum((y_true==1) & (thresholded_probs==0))
  return FN



"""
    Get accuracy of predictions given a threshold for classification.
"""



def get_accuracy(y_true, y_pred_prob, th):
  TP = true_positives(y_true, y_pred_prob, th)
  TN = true_negatives(y_true, y_pred_prob, th)
  FP = false_positives(y_true, y_pred_prob, th)
  FN = false_negatives(y_true, y_pred_prob, th)
  accuracy = (TP + TN)/(TP + TN + FP + FN)
  return accuracy



"""
    Get prevalence of class '1'. Class '1' here refers to COVID-19 positive cases.
"""



def get_prevalence(y_true):
  prevalence = 0
  prevalence = np.sum(y_true)/len(y_true)
  return prevalence



"""
    Get sensitivity.
    Sensitivity is the conditional probability that the model 
    predicts true given the label is actually true.
"""



def get_sensitivity(y_true, y_pred_prob, th):
  sensitivity = 0
  TP = true_positives(y_true, y_pred_prob, th)
  FN = false_negatives(y_true, y_pred_prob, th)
  sensitivity = TP / (TP + FN)
  return sensitivity



"""
    Get specificity.
    Specificity is the conditional probability that the model 
    predicts false given the label is actually false.
"""



def get_specificity(y_true, y_pred_prob, th):
  specificity = 0
  TN = true_negatives(y_true, y_pred_prob, th)
  FP = false_positives(y_true, y_pred_prob, th)
  specificity = TN / (TN + FP)
  return specificity



"""
    Get Positive Predictive Value.
    PPV is the conditional probability that the label is actually 
    true given the model predicts true.
"""



def get_ppv(y_true, y_pred_prob, th):
  ppv = 0
  TP = true_positives(y_true, y_pred_prob, th)
  FP = false_positives(y_true, y_pred_prob, th)
  ppv = TP / (FP + TP)
  return ppv



"""
    Get Negative Predictive Value.
    NPV is the conditional probability that the label is actually 
    false given the model predicts false. 
"""



def get_npv(y_true, y_pred_prob, th):
  npv = 0
  TN = true_negatives(y_true, y_pred_prob, th)
  FN = false_negatives(y_true, y_pred_prob, th)
  npv = TN / (FN + TN)
  return npv



"""
    Get ROC curve.
"""



def get_roc_curve_info(y_true, y_pred_prob):
  threshold_set = np.arange(0,1,0.00001)
  TPR = np.zeros(len(threshold_set))
  FPR = np.zeros(len(threshold_set))
  i=0
  for th in threshold_set:
    TPR[i] = get_sensitivity(y_true, y_pred_prob, th)
    spec = get_specificity(y_true, y_pred_prob, th)
    FPR[i] = (1 - spec)
    i+=1
  return TPR, FPR



"""
    Get Precision-Recall curve.
"""



def precision_recall_curve_info(y_true, y_pred_prob):
  threshold_set = np.arange(0,1,0.00001)
  precision = np.zeros(len(threshold_set))
  recall = np.zeros(len(threshold_set))
  i=0
  for th in threshold_set:
    precision[i] = get_ppv(y_true, y_pred_prob, th)
    recall[i] = get_sensitivity(y_true, y_pred_prob, th)
    i+=1
  precision = np.array([i for i in precision if (str(i)!='nan')])
  precision = np.concatenate((precision, np.ones(len(recall) - len(precision))))
  return precision, recall




"""
    Call previous functions from get_statistics().
"""



def get_statistics(y_true, y_pred_prob, th):
  accuracy = get_accuracy(y_true, y_pred_prob, th)
  prevalence = get_prevalence(y_true)
  sensitivity = get_sensitivity(y_true, y_pred_prob, th)
  specificity = get_specificity(y_true, y_pred_prob, th)
  ppv = get_ppv(y_true, y_pred_prob, th)
  npv = get_npv(y_true, y_pred_prob, th)
  tpr, fpr = get_roc_curve_info(y_true, y_pred_prob)
  AUC = auc(fpr,tpr)
  prec_list, rec_list = precision_recall_curve_info(y_true, y_pred_prob)
  score = auc(rec_list, prec_list)
  f1_score = 2*ppv*sensitivity/(ppv + sensitivity)
  stat_dict = {'Accuracy': accuracy, 
               'Prevalence':prevalence,
               'Sensitivity':sensitivity,
               'Specificity':specificity,
               'PPV':ppv,
               'NPV':npv,
               'AUC':AUC,
               'F1 Score':f1_score,
               'Score': score}
  return stat_dict, tpr, fpr, prec_list, rec_list




"""
    Function for deciding optimal threshold for classifier.
    Makes use of the ROC to decide a good threshold.
"""



def decide_th(y_true, y_pred_prob):
  threshold_set = np.arange(0,1,0.00001)
  TPR = np.zeros(len(threshold_set))
  FPR = np.zeros(len(threshold_set))
  J = np.zeros(len(threshold_set))
  i=0
  for th in threshold_set:
    TPR[i] = get_sensitivity(y_true, y_pred_prob, th)
    spec = get_specificity(y_true, y_pred_prob, th)
    FPR[i] = (1 - spec)
    J[i] = TPR[i] - FPR[i]
    i+=1
  good_th = np.argmax(J)
  return threshold_set[good_th]



"""
    Functions for plotting.
"""



def plot_prc(precision, recall):
  plt.plot(recall, precision, 'r')
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision-Recall Curve")
  plt.show()

def plot_roc(TPR, FPR):
  plt.plot(FPR, TPR, 'r', label = "ROC Curve")
  plt.plot(np.arange(0,1.05,0.05), np.arange(0,1.05,0.05), 'k', label = "y=x line")
  plt.legend(loc = "lower right")
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.title("ROC Curve")
  plt.show()