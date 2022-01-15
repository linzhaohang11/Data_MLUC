import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from urllib.parse import urljoin

dic = globals()

Set_gen = [3, 4, 7, 8, 20, 21, 30, 31]

# Get all data from root path
url_root = 'https://raw.githubusercontent.com/linzhaohang11/Data_MLUC/main/Bus_24/Data/'
Data_feature_path = urljoin(url_root, 'Data_feature.csv')
Data_feature = pd.read_csv(Data_feature_path)

# Add index for feature data
RES_01 = list(range(1, 25))
RES_02 = list(range(1, 25))
RES_03 = list(range(1, 25))
RES_04 = list(range(1, 25))
RES_05 = list(range(1, 25))
Load   = list(range(1, 25))

for t in range(1,25):
    RES_01[t - 1] = 'RES_01_H' + str(t)
    RES_02[t - 1] = 'RES_02_H' + str(t)
    RES_03[t - 1] = 'RES_03_H' + str(t)
    RES_04[t - 1] = 'RES_04_H' + str(t)
    RES_05[t - 1] = 'RES_05_H' + str(t)
    Load[t - 1]   = 'Load_H'   + str(t)

Feature_index = RES_01 + RES_02 + RES_03 + RES_04 + RES_05 + Load
Data_feature.index = Feature_index

# Add index for the label data
Hour = list(range(1, 25))
for t in range(1, 25):
    Hour[t - 1] = 'Hour' + str(t)

for i_gen in Set_gen:
    if i_gen < 10:
        dic['Data_label_G0' + str(i_gen) + '_path'] = urljoin(url_root, 'Data_label_G0' + str(i_gen) + '.csv')
        dic['Data_label_G0' + str(i_gen)] = pd.read_csv(dic['Data_label_G0' + str(i_gen) + '_path'])
        dic['Data_label_G0' + str(i_gen)].index = Hour
        dic['Data_label_G0' + str(i_gen)] = dic['Data_label_G0' + str(i_gen)].drop('Unnamed: 0', axis=1)
    if i_gen >= 10:
        dic['Data_label_G'  + str(i_gen) + '_path'] = urljoin(url_root, 'Data_label_G' + str(i_gen) + '.csv')
        dic['Data_label_G'  + str(i_gen)] = pd.read_csv(dic['Data_label_G'  + str(i_gen) + '_path'])
        dic['Data_label_G'  + str(i_gen)].index = Hour
        dic['Data_label_G' + str(i_gen)] = dic['Data_label_G' + str(i_gen)].drop('Unnamed: 0', axis=1)

# Get train and test data
for i_gen in Set_gen:
    if i_gen < 10:
        dic['Train_label_G0'   + str(i_gen)] = dic['Data_label_G0' + str(i_gen)].loc[:,'Day_0001':'Day_3648']
        dic['Test_label_G0'  + str(i_gen)] = dic['Data_label_G0' + str(i_gen)].loc[:  ,'Day_3649':'Day_4384']
    if i_gen >= 10:
        dic['Train_label_G'   + str(i_gen)] = dic['Data_label_G' + str(i_gen)].loc[:,'Day_0001':'Day_3648']
        dic['Test_label_G'  + str(i_gen)] = dic['Data_label_G' + str(i_gen)].loc[:  ,'Day_3649':'Day_4384']

Train_feature = Data_feature.loc[:,'Day_0001':'Day_3648']
Test_feature  = Data_feature.loc[:,'Day_3649':'Day_4384']

# Do Grid Search for identifying the best parameters
param_grid = {'max_depth':[5,6,7,8,9], 'min_samples_split': [2, 4], 'min_samples_leaf': [2, 4], 'max_leaf_nodes': [5, 25]}
for i_gen in Set_gen:
    if i_gen < 10:
        dic['DT_G0' + str(i_gen)] = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
        dic['DT_GS_G0' + str(i_gen)] = GridSearchCV(dic['DT_G0' + str(i_gen)], param_grid, cv = 10, scoring = 'accuracy')
        dic['DT_GS_G0' + str(i_gen)].fit(Train_feature.values.T, dic['Train_label_G0' + str(i_gen)].values.T)
        dic['Best_Para_G0' + str(i_gen)] = dic['DT_GS_G0' + str(i_gen)].best_params_
        print('Parameters for DT_G0',str(i_gen), ':', dic['Best_Para_G0' + str(i_gen)])
    if i_gen >= 10:
        dic['DT_G' + str(i_gen)] = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
        dic['DT_GS_G' + str(i_gen)] = GridSearchCV(dic['DT_G' + str(i_gen)], param_grid, cv = 10, scoring = 'accuracy')
        dic['DT_GS_G' + str(i_gen)].fit(Train_feature.values.T, dic['Train_label_G' + str(i_gen)].values.T)
        dic['Best_Para_G' + str(i_gen)] = dic['DT_GS_G' + str(i_gen)].best_params_
        print('Parameters for DT_G',str(i_gen), ':',  dic['Best_Para_G' + str(i_gen)])

# Use the best parameters to design trees
for i_gen in Set_gen:
    if i_gen < 10:
        dic['DT_G0' + str(i_gen)] = tree.DecisionTreeClassifier(criterion         = 'gini',
                                                                splitter          = 'best',
                                                                max_depth         = dic['Best_Para_G0' + str(i_gen)]['max_depth'],
                                                                min_samples_split = dic['Best_Para_G0' + str(i_gen)]['min_samples_split'],
                                                                min_samples_leaf  = dic['Best_Para_G0' + str(i_gen)]['min_samples_leaf'],
                                                                max_leaf_nodes    = dic['Best_Para_G0' + str(i_gen)]['max_leaf_nodes'])
        dic['DT_G0' + str(i_gen)] = dic['DT_G0' + str(i_gen)].fit(Train_feature.T, dic['Train_label_G0' + str(i_gen)].T)
    if i_gen >= 10:
        dic['DT_G' + str(i_gen)] = tree.DecisionTreeClassifier(criterion         = 'gini',
                                                               splitter          = 'best',
                                                               max_depth         = dic['Best_Para_G' + str(i_gen)]['max_depth'],
                                                               min_samples_split = dic['Best_Para_G' + str(i_gen)]['min_samples_split'],
                                                               min_samples_leaf  = dic['Best_Para_G' + str(i_gen)]['min_samples_leaf'],
                                                               max_leaf_nodes    = dic['Best_Para_G' + str(i_gen)]['max_leaf_nodes'])
        dic['DT_G' + str(i_gen)] = dic['DT_G' + str(i_gen)].fit(Train_feature.T, dic['Train_label_G' + str(i_gen)].T)

# Check the scores
for i_gen in Set_gen:
    if i_gen < 10:
        dic['DT_pred_G0' + str(i_gen)] = dic['DT_G0' + str(i_gen)].predict(Test_feature.T).T
        dic['DT_true_G0' + str(i_gen)] = np.array(dic['Test_label_G0' + str(i_gen)])
        print('-----------------------------------------------------------------------------------------------')
        print('Scores for G0', str(i_gen))
        print(classification_report(dic['DT_true_G0' + str(i_gen)].flatten(), dic['DT_pred_G0' + str(i_gen)].flatten()))
        if len(np.unique(dic['DT_true_G0' + str(i_gen)])) > 1:
            print('ROC_score         ', roc_auc_score(dic['DT_true_G0' + str(i_gen)].flatten(), dic['DT_pred_G0' + str(i_gen)].flatten()))
        else:
            print('Only one class, no ROC score.')
            print('-----------------------------------------------------------------------------------------------')
    if i_gen >= 10:
        dic['DT_pred_G' + str(i_gen)] = dic['DT_G' + str(i_gen)].predict(Test_feature.T).T
        dic['DT_true_G' + str(i_gen)] = np.array(dic['Test_label_G' + str(i_gen)])
        print('-----------------------------------------------------------------------------------------------')
        print('Scores for G', str(i_gen))
        print(classification_report(dic['DT_true_G' + str(i_gen)].flatten(), dic['DT_pred_G' + str(i_gen)].flatten()))
        if len(np.unique(dic['DT_true_G' + str(i_gen)])) > 1:
            print('ROC_score         ', roc_auc_score(dic['DT_true_G' + str(i_gen)].flatten(), dic['DT_pred_G' + str(i_gen)].flatten()))
        else:
            print('Only one class, no ROC score.')
            print('-----------------------------------------------------------------------------------------------')