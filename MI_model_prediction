import numpy as np
import pandas as pd
import scipy as sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import  roc_auc_score, precision_score, recall_score, confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

#Importing the data 
data_MI_inci = pd.read_csv("UKB_MI_clinical_FD_dataset.csv")

#Defining the variables
y = data_MI_inci["MI_block"]
X= data_MI_inci.drop(["IID","MI_block"], axis=1)


#Train and test the MI model. Testing implies using 10 fold CV and using the same partinioning across all the models 
cv_outer = StratifiedKFold(n_splits=10)
outer_results_score = list()
outer_results_rfc_Df = list()
outer_results_rfc_DfPRS = list()
outer_results_rfc_PRS = list()

counter= 0
for train_idx, val_idx in cv_outer.split(X, y):
    train_data, val_data = X.iloc[train_idx], X.iloc[val_idx]
    train_target, val_target = y.iloc[train_idx], y.iloc[val_idx]
    
    #SCORE MODEL transformation
    val_data_SCORE = val_data.drop(["PRS","Fd_left", "Fd_right", "age", "SBP_device", "BMI"], axis=1)
    val_data_SCORE = pd.get_dummies(val_data_SCORE, columns=["age_cat", "SBP_cat", "BMI_cat"])
    
    train_data_SCORE = train_data.drop(["PRS","Fd_left", "Fd_right", "age", "SBP_device", "BMI"], axis=1)
    train_data_SCORE = pd.get_dummies(train_data_SCORE, columns=["age_cat", "SBP_cat", "BMI_cat"])
    
    # Get missing columns in the validation test
    missing_cols = set(X_base.columns ) - set( val_data_SCORE.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        val_data_SCORE[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    val_data_SCORE = val_data_SCORE[X_base.columns]
    
    # Get missing columns in the training test
    missing_cols = set(X_base.columns ) - set( train_data_SCORE.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        train_data_SCORE[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    train_data_SCORE = train_data_SCORE[X_base.columns]

    #SCORE model and predictions
    model_SCORE = LogisticRegression(n_jobs=-1)
    baseline_model = model_SCORE.fit(train_data_SCORE, train_target)
    val_dat_hat = baseline_model.predict(val_data_SCORE)
    
    y_SCORE_hat = baseline_model.predict_proba(val_data_SCORE)
    y_SCORE_hat = pd.DataFrame(y_SCORE_hat)
    y_1 = y_SCORE_hat[[1]]
    
    t_data  = val_target.tolist()
    t_data = pd.DataFrame(t_data)
    
    index_names = val_target.index.to_list()
    index_names = pd.DataFrame(index_names)

    prec_score = precision_score(val_target, val_dat_hat)
    rec_score = recall_score(val_target, val_dat_hat)
    auc = roc_auc_score(val_target, val_dat_hat)
    outer_results_score.append([prec_score, rec_score, auc])
    
    #RFC model introducing FRACTAL DIMENSION
    val_data_2 = val_data.drop(["PRS", "age_cat", "BMI_cat", "SBP_cat"], axis=1)
    train_data_inci_2 =train_data.drop(["PRS", "age_cat", "BMI_cat", "SBP_cat"], axis=1)
    
    model_rfc_2 = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_depth=6)
    MI_rfc_2 = model_rfc_2.fit(train_data_inci_2, train_target)
    y_2_hat_rfc = MI_rfc_2.predict(val_data_2)
    
    y_2_hat = MI_rfc_2.predict_proba(val_data_2)
    y_2_hat = pd.DataFrame(y_2_hat)
    y_2 = y_2_hat[[1]]
    
    prec_rfc_2 = precision_score(val_target, y_2_hat_rfc)
    rec_rfc_2 = recall_score(val_target, y_2_hat_rfc)
    auc_rfc_2= roc_auc_score(val_target, y_2_hat_rfc)
    outer_results_rfc_Df.append([prec_rfc_2, rec_rfc_2, auc_rfc_2])
    
    #RFC WITH Df and CAD PRS 
    train_data_3 =train_data.drop(["age_cat", "BMI_cat", "SBP_cat"], axis=1)
    val_data_3 = val_data.drop(["age_cat", "BMI_cat", "SBP_cat"], axis=1)
    
    model_rfc_3 = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_depth=6)
    MI_rfc_3 = model_rfc_3.fit(train_data_3, train_target)
    y_hat_rfc_3 = MI_rfc_3.predict(val_data_3)
    
    y_3_hat = MI_rfc_3.predict_proba(val_data_3)
    y_3_hat = pd.DataFrame(y_3_hat)
    y_3 = y_3_hat[[1]]
    
    prec_rfc_3 = precision_score(val_target, y_hat_rfc_3)
    rec_rfc_3 = recall_score(val_target, y_hat_rfc_3)
    auc_rfc_3 = roc_auc_score(val_target, y_hat_rfc_3)
    outer_results_rfc_DfPRS.append([prec_rfc_3, rec_rfc_3, auc_rfc_3])
    
    #RFC model introducing CAD PRS
    val_data_4 = val_data.drop(["Fd_left", "Fd_right", "age_cat", "BMI_cat", "SBP_cat"], axis=1)
    train_data_4 =train_data.drop(["Fd_left", "Fd_right", "age_cat", "BMI_cat", "SBP_cat"], axis=1)
    
    model_rfc_4 = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_depth=6)
    MI_rfc_4 = model_rfc_4.fit(train_data_4, train_target)
    y_hat_rfc_4 = MI_rfc_4.predict(val_data_4)
    
    y_4_hat = MI_rfc_4.predict_proba(val_data_4)
    y_4_hat = pd.DataFrame(y_4_hat)
    y_4 = y_4_hat[[1]]
    
    prec_rfc_4 = precision_score(val_target, y_hat_rfc_4)
    rec_rfc_4 = recall_score(val_target, y_hat_rfc_4)
    auc_rfc_4 = roc_auc_score(val_target, y_hat_rfc_4)
    outer_results_rfc_PRS.append([prec_rfc_4, rec_rfc_4, auc_rfc_4])
     
    
    #### Merging the predictions of all models for each fold ans save them
    fold_predictions = pd.concat([index_names, t_data, y_1, y_2, y_3, y_4], axis=1)
    name = "MI_models_predictions_F" + str(counter) + ".csv"
    fold_predictions.to_csv(name, header=None, index=None)
    counter += 1
    
    
outer_results_score = pd.DataFrame(outer_results_score)
outer_results_score.columns = ["Precision", "Recall", "AUC"]
outer_results_score.describe()

outer_results_rfc_Df = pd.DataFrame(outer_results_rfc_Df)
outer_results_rfc_Df.columns = ["Precision", "Recall", "AUC"]
outer_results_rfc_Df.describe()

outer_results_rfc_DfPRS = pd.DataFrame(outer_results_rfc_DfPRS)
outer_results_rfc_DfPRS.columns = ["Precision", "Recall", "AUC"]
outer_results_rfc_DfPRS.describe()

outer_results_rfc_PRS = pd.DataFrame(outer_results_rfc_PRS)
outer_results_rfc_PRS.columns = ["Precision", "Recall", "AUC"]
outer_results_rfc_PRS.describe()



sci.stats.wilcoxon(outer_results_rfc_Df['AUC'], y=outer_results_score['AUC'], zero_method='zsplit', correction=False, alternative='two-sided', mode='exact')
sci.stats.wilcoxon(outer_results_rfc_DfPRS['AUC'], y=outer_results_score['AUC'], zero_method='zsplit', correction=False, alternative='two-sided', mode='exact')
sci.stats.wilcoxon(outer_results_rfc_PRS['AUC'], y=outer_results_score['AUC'], zero_method='zsplit', correction=False, alternative='two-sided', mode='exact')
sci.stats.wilcoxon(outer_results_rfc_Df['AUC'], y=outer_results_rfc_DfPRS['AUC'], zero_method='zsplit', correction=False, alternative='two-sided', mode='exact')
sci.stats.wilcoxon(outer_results_rfc_Df['AUC'], y=outer_results_rfc_PRS['AUC'], zero_method='zsplit', correction=False, alternative='two-sided', mode='exact')
sci.stats.wilcoxon(outer_results_rfc_DfPRS['AUC'], y=outer_results_rfc_PRS['AUC'], zero_method='zsplit', correction=False, alternative='two-sided', mode='exact')

#Obtaining the file with the predictions of all models 
F0 = pd.read_csv("MI_models_predictions_F0.csv", header = None)
F1 = pd.read_csv("MI_models_predictions_F1.csv", header = None)
F2 = pd.read_csv("MI_models_predictions_F2.csv", header = None)
F3 = pd.read_csv("MI_models_predictions_F3.csv", header = None)
F4 = pd.read_csv("MI_models_predictions_F4.csv", header = None)
F5 = pd.read_csv("MI_models_predictions_F5.csv", header = None)
F6 = pd.read_csv("MI_models_predictions_F6.csv", header = None)
F7 = pd.read_csv("MI_models_predictions_F7.csv", header = None)
F8 = pd.read_csv("MI_models_predictions_F8.csv", header = None)
F9 = pd.read_csv("MI_models_predictions_F9.csv", header = None)
pred_results_MI = pd.concat([F0,F1, F2,F3, F4, F5, F6, F7, F8, F9], axis=0)
pred_results_MI.columns = ['index','True_data', 'SCORE', 'RFC_Df', 'RFC_DfPRS', 'RFC_PRS']
pred_results_MI.to_csv("MI_models_predictions_10FCV.csv", index=None)




