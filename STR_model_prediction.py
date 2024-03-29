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

#load the STR data
data_all = pd.read_csv("Stroke_BN_I_Df_dataset.csv")

# Defining independent and dependent variables
y = data_raw["STR_anyBN_I"]
X= data_raw.drop(["IID", "STR_anyBN_I"], axis=1)

#Train and test the STR model. Testing implies using 10 fold CV and using the same partinioning across all the models 
cv_outer = StratifiedKFold(n_splits=10)
outer_results_base = list()
outer_results_rfc_Df = list()
outer_results_rfc_DfPRS = list()
outer_results_rfc_PRS = list()

counter= 0
for train_idx, val_idx in cv_outer.split(X, y):
    train_data, val_data = X.iloc[train_idx], X.iloc[val_idx]
    train_target, val_target = y.iloc[train_idx], y.iloc[val_idx]
    
    #Baseline transformation
    val_data_base = val_data.drop(["Fd_left", "Fd_right", "PRS"], axis=1)
    train_data_base = train_data.drop(["Fd_left", "Fd_right", "PRS"], axis=1)

    #SCORE model and predictions
    model_base = LogisticRegression(n_jobs=-1)
    baseline_model = model_base.fit(train_data_base, train_target)
    val_dat_hat = baseline_model.predict(val_data_base)
    
    y_base_hat = baseline_model.predict_proba(val_data_base)
    y_base_hat = pd.DataFrame(y_base_hat)
    y_1 = y_base_hat[[1]]
    
    t_data  = val_target
    t_data = pd.DataFrame(t_data)
    t_data['index_name'] = val_target.index
    t_data['base'] = y_base_hat[1].values

    prec_score = precision_score(val_target, val_dat_hat)
    rec_score = recall_score(val_target, val_dat_hat)
    auc = roc_auc_score(val_target, val_dat_hat)
    outer_results_base.append([prec_score, rec_score, auc])
    
    #RFC model introducing FRACTAL DIMENSION AND STR PRS
    val_data_2 = val_data
    train_data_2= train_data
    
    model_rfc_2 = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_depth=6)
    
    STR_rfc_2 = model_rfc_2.fit(train_data_2, train_target)
    y_2_hat_rfc = STR_rfc_2.predict(val_data_2)
    
    y_2_hat = STR_rfc_2.predict_proba(val_data_2)
    y_2_hat = pd.DataFrame(y_2_hat)
    y_2 = y_2_hat[[1]]
    t_data['RFC_DfPRS'] = y_2_hat[1].values
    
    prec_rfc_2 = precision_score(val_target, y_2_hat_rfc)
    rec_rfc_2 = recall_score(val_target, y_2_hat_rfc)
    auc_rfc_2= roc_auc_score(val_target, y_2_hat_rfc)
    outer_results_rfc_DfPRS.append([prec_rfc_2, rec_rfc_2, auc_rfc_2])
    
    #RFC model introducing FRACTAL DIMENSION
    val_data_3 = val_data.drop(["PRS"], axis=1)
    train_data_3= train_data.drop(["PRS"], axis=1)
    
    STR_rfc_3 = model_rfc_2.fit(train_data_3, train_target)
    y_3_hat_rfc = STR_rfc_3.predict(val_data_3)
    
    y_3_hat = STR_rfc_3.predict_proba(val_data_3)
    y_3_hat = pd.DataFrame(y_3_hat)
    y_3 = y_3_hat[[1]]
    t_data['RFC_Df'] = y_3_hat[1].values
    
    prec_rfc_3 = precision_score(val_target, y_3_hat_rfc)
    rec_rfc_3 = recall_score(val_target, y_3_hat_rfc)
    auc_rfc_3= roc_auc_score(val_target, y_3_hat_rfc)
    outer_results_rfc_Df.append([prec_rfc_3, rec_rfc_3, auc_rfc_3])
    
    #RFC model introducing STROKE PRS
    
    val_data_4= val_data.drop(["Fd_left", "Fd_right"], axis=1)
    train_data_4= train_data.drop(["Fd_left", "Fd_right"], axis=1)
    val_data_4 = val_data
    train_data_4= train_data
    
    STR_rfc_4 = model_rfc_2.fit(train_data_4, train_target)
    y_4_hat_rfc = STR_rfc_4.predict(val_data_4)
    
    y_4_hat = STR_rfc_4.predict_proba(val_data_4)
    y_4_hat = pd.DataFrame(y_4_hat)
    y_4 = y_4_hat[[1]]
    t_data['RFC_PRS'] = y_4_hat[1].values
    
    prec_rfc_4 = precision_score(val_target, y_4_hat_rfc)
    rec_rfc_4 = recall_score(val_target, y_4_hat_rfc)
    auc_rfc_4= roc_auc_score(val_target, y_4_hat_rfc)
    outer_results_rfc_PRS.append([prec_rfc_4, rec_rfc_4, auc_rfc_4])
    
    
    #### Merging the predictions of all models for each fold ans save them
    name = "STR_AnyBN_models_predictions_F" + str(counter) + ".csv"
    t_data.to_csv(name, header=None, index=None)
    counter += 1

outer_results_base = pd.DataFrame(outer_results_base)
outer_results_base.columns = ["Precision", "Recall", "AUC"]
outer_results_base.describe()

outer_results_rfc_Df = pd.DataFrame(outer_results_rfc_Df)
outer_results_rfc_Df.columns = ["Precision", "Recall", "AUC"]
outer_results_rfc_Df.describe()

outer_results_rfc_DfPRS = pd.DataFrame(outer_results_rfc_DfPRS)
outer_results_rfc_DfPRS.columns = ["Precision", "Recall", "AUC"]
outer_results_rfc_DfPRS.describe()

outer_results_rfc_PRS = pd.DataFrame(outer_results_rfc_PRS)
outer_results_rfc_PRS.columns = ["Precision", "Recall", "AUC"]
outer_results_rfc_PRS.describe()

sci.stats.wilcoxon(outer_results_rfc_Df['AUC'], y=outer_results_base['AUC'], zero_method='zsplit', correction=False)
sci.stats.wilcoxon(outer_results_rfc_DfPRS['AUC'], y=outer_results_base['AUC'], zero_method='zsplit', correction=False)
sci.stats.wilcoxon(outer_results_rfc_PRS['AUC'], y=outer_results_base['AUC'], zero_method='zsplit', correction=False)
sci.stats.wilcoxon(outer_results_rfc_PRS['AUC'], y=outer_results_base['AUC'], zero_method='zsplit', correction=False)
sci.stats.wilcoxon(outer_results_rfc_Df['AUC'], y=outer_results_rfc_DfPRS['AUC'], zero_method='zsplit', correction=False)
sci.stats.wilcoxon(outer_results_rfc_Df['AUC'], y=outer_results_rfc_PRS['AUC'], zero_method='zsplit', correction=False)
sci.stats.wilcoxon(outer_results_rfc_DfPRS['AUC'], y=outer_results_rfc_PRS['AUC'], zero_method='zsplit', correction=False)


#Obtaining the file with the predictions of all models 
F0 = pd.read_csv("STR_AnyBN_models_predictions_F0.csv", header = None)
F1 = pd.read_csv("STR_AnyBN_models_predictions_F1.csv", header = None)
F2 = pd.read_csv("STR_AnyBN_models_predictions_F2.csv", header = None)
F3 = pd.read_csv("STR_AnyBN_models_predictions_F3.csv", header = None)
F4 = pd.read_csv("STR_AnyBN_models_predictions_F4.csv", header = None)
F5 = pd.read_csv("STR_AnyBN_models_predictions_F5.csv", header = None)
F6 = pd.read_csv("STR_AnyBN_models_predictions_F6.csv", header = None)
F7 = pd.read_csv("STR_AnyBN_models_predictions_F7.csv", header = None)
F8 = pd.read_csv("STR_AnyBN_models_predictions_F8.csv", header = None)
F9 = pd.read_csv("STR_AnyBN_models_predictions_F9.csv", header = None)
pred_results_STR = pd.concat([F0,F1, F2,F3, F4, F5, F6, F7, F8, F9], axis=0)
pred_results_STR.columns = ['True_data', 'index', 'SCORE', 'RFC_Df', 'RFC_DfPRS', 'RFC_PRS']
pred_results_STR.to_csv("STR_BN_models_predictions_10FCV_complete.csv", index=None)
