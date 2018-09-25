# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from splitdf import split_df
from info_value import iv
# from .info_ent_indx_gini import (ig, ie)
from var_filter import var_filter
from woebin import (woebin, woebin_ply, woebin_plot, woebin_adj)
from perf import (perf_eva, perf_psi)
from scorecard import (scorecard, scorecard_ply)
import pandas as pd

# Logistic Regression

dat = pd.read_csv('data/creditdataset.csv')

ivlist=iv(dat, y="creditability")
ivlist


# filter variable via missing rate, iv, identical value rate
dt_s = var_filter(dat, y="creditability", iv_limit=0.02)

# breaking dt into train and test
train, test = split_df(dt_s, 'creditability', ratio = [0.7, 0.3]).values()

# woe binning ------

bins = woebin(dt_s, y="creditability")
#woebin_plot(bins)

# binning adjustment
# # adjust breaks interactively
# breaks_adj = woebin_adj(dt_s, "creditability", bins) 
# # or specify breaks manually
breaks_adj = {
    'age.in.years': [27, 35, 45, 55],
    'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
}
bins_adj = woebin(dt_s, y="creditability", breaks_list=breaks_adj)


# converting train and test into woe values
train_woe = woebin_ply(train, bins_adj)
test_woe = woebin_ply(test, bins_adj)

y_train = train_woe.loc[:,'creditability']
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']

# logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_

# predicted proability
train_pred = lr.predict_proba(X_train)[:,1]
test_pred = lr.predict_proba(X_test)[:,1]

# performance ks & roc ------
train_perf = perf_eva(y_train, train_pred, title = "train")
test_perf = perf_eva(y_test, test_pred, title = "test")

#perf_eva(y_test, test_pred, plot_type = ["ks","lift","roc","pr"])


# score ------
card = scorecard(bins_adj, lr, X_train.columns)

# credit score
train_score = scorecard_ply(train, card, print_step=0)
test_score = scorecard_ply(test, card, print_step=0)


# psi
perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)



from sklearn import metrics

y_pred_df = pd.DataFrame( { 'actual': y_test, "predicted_prob": test_pred } )
y_pred_df['predicted'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.5 else 0)

import matplotlib.pylab as plt

# Confusion Matrix
import seaborn as sn

def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix( actual, predicted, [1,0] )
    sn.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Default", "No Default"] , yticklabels = ["Default", "No Default"] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

draw_cm( y_pred_df.actual, y_pred_df.predicted )


# Overall accuracy of the model
import numpy as np

print( 'Total Accuracy : ',np.round( metrics.accuracy_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Precision : ',np.round( metrics.precision_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Recall : ',np.round( metrics.recall_score( y_test, y_pred_df.predicted ), 2 ) )

cm1 = metrics.confusion_matrix( y_pred_df.actual, y_pred_df.predicted, [1,0] )

sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', round( sensitivity, 2) )

specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', round( specificity, 2 ) )

# Predicted Probability distribution Plots for Defaults and Non Defaults
#sn.distplot( y_pred_df[y_pred_df.actual == 1]["predicted_prob"], kde=False, color = 'b' )
#sn.distplot( y_pred_df[y_pred_df.actual == 0]["predicted_prob"], kde=False, color = 'g' )


auc_score = metrics.roc_auc_score( y_pred_df.actual, y_pred_df.predicted_prob  )
round( float( auc_score ), 2 )

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds

fpr, tpr, thresholds = draw_roc( y_pred_df.actual, y_pred_df.predicted_prob )

# Finding Optimal Cutoff Probability
thresholds[0:10]
fpr[0:10]
tpr[0:10]


# Find optimal cutoff using youden's index
# Youden's index is where (Sensitivity+Specificity - 1) is maximum.
# That is when (TPR+TNR -1) is maximum.
#    max( TPR - (1 - TNR) )
#    max( TPR - FPR )

tpr_fpr = pd.DataFrame( { 'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds } )
tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr.sort_values( 'diff', ascending = False )[0:10]

y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.3 else 0)
draw_cm( y_pred_df.actual, y_pred_df.predicted_new )

#
## Find optimal cutoff probability using cost
#
#cm = metrics.confusion_matrix( y_pred_df.actual, y_pred_df.predicted_new, [1,0] )
#cm_mat = np.array( cm )
#cm_mat[1, 0]
#cm_mat[0, 1]
#
#def get_total_cost( actual, predicted ):
#    cm = metrics.confusion_matrix( actual, predicted, [1,0] )
#    cm_mat = np.array( cm )
#    return cm_mat[0,1] * 2 + cm_mat[1,0] * 1
#
#get_total_cost( y_pred_df.actual, y_pred_df.predicted_new )
#
#cost_df = pd.DataFrame( columns = ['prob', 'cost'])
#
#idx = 0
#for each_prob in range( 20, 50):
#    cost = get_total_cost( y_pred_df.actual,
#                          y_pred_df.predicted_prob.map(
#            lambda x: 1 if x > (each_prob/100)  else 0) )
#    cost_df.loc[idx] = [(each_prob/100), cost]
#    idx += 1
#
#cost_df.sort_values( 'cost', ascending = True )[0:5]
#
#y_pred_df['predicted_final'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.20 else 0)
#draw_cm( y_pred_df.actual, y_pred_df.predicted_final )
#
#print( 'Total Accuracy : ',np.round( metrics.accuracy_score( y_test, y_pred_df.predicted_final ), 2 ) )
#print( 'Precision : ',np.round( metrics.precision_score( y_test, y_pred_df.predicted_final ), 2 ) )
#print( 'Recall : ',np.round( metrics.recall_score( y_test, y_pred_df.predicted_final ), 2 ) )
#
#cm1 = metrics.confusion_matrix( y_pred_df.actual, y_pred_df.predicted_final, [1,0] )
#
#sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
#print('Sensitivity : ', round( sensitivity, 2) )
#
#specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
#print('Specificity : ', round( specificity, 2 ) )
#
## Total accuracy of the model is 67%, becuase the objective is not to improve total accuracy but minimize the quadrants that contribute to the cost.
