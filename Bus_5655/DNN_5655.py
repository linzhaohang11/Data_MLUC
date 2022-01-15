import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from urllib.parse import urljoin

dic = globals()

Set_gen = [1  ,2  ,3  ,4  ,5  ,6  ,7  ,8  ,9  ,10 ,13 ,14 ,15 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,
           33,34,35,36,37,38,39,40,41,42,43,47,50 ,73 ,77 ,78 ,84 ,85 ,86 ,87 ,96 ,97 ,101,102,105,106,108,109,115,117,
           121,122,123,124,125,131,132,133,134,135,136,140,141,143,146,147,148,150,153,154,155,159,160,161,162,165,166,
           171,172,173,178,179,180,181,182,183,190,191,192,193,195,196,197,198,199,200,201,203,204,205,206,207,208,211,
           212,215,216,217,218,219,220,221,222,223,226,227,228,229,230,231,232,235,236,237,238,239,240,241,242,243,244,
           245,246,261,265,266,282,283,284,285,286,287,288,289,297,302,303,304,305,306,307,308,309,311,312,313,314,315,
           316,317,318,319,320,321,322,323,324,329,344,348,349,355,356,357,358,359,360,361,362,366,379,380,381,382,383,
           384,385,390,391,402,409,410,411,412,413,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,
           433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,450,451,453,454,455,456,457,458,459,460,461]

# Get all the data from root path
url_root = 'https://raw.githubusercontent.com/linzhaohang11/Data_MLUC/main/Bus_5655/Data/'
Data_feature_path = urljoin(url_root, 'Data_feature.csv')
Data_feature = pd.read_csv(Data_feature_path)/10000

# Add index for the feature data
RES_01 = list(range(1,25))
RES_02 = list(range(1,25))
RES_03 = list(range(1,25))
RES_04 = list(range(1,25))
RES_05 = list(range(1,25))
Load   = list(range(1,25))

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
Hour = list(range(1,25))
for t in range(1,25):
    Hour[t - 1] = 'Hour' + str(t)

# Get label
Data_label_G_path = urljoin(url_root, 'Data_label_G.csv')
Data_label_G = pd.read_csv(Data_label_G_path)
Data_label_G = Data_label_G.drop('Unnamed: 0', axis=1)
Data_label_G.columns

for i_gen in Set_gen:
    if i_gen < 10:
        dic['Data_label_G00' + str(i_gen) + '_path'] = urljoin(url_root, 'Data_label_G00' + str(i_gen) + '.csv')
        dic['Data_label_G00' + str(i_gen)] = np.genfromtxt(dic['Data_label_G00' + str(i_gen) + '_path'], delimiter=',')
        dic['Data_label_G00' + str(i_gen)] = pd.DataFrame(dic['Data_label_G00' + str(i_gen)])
        dic['Data_label_G00' + str(i_gen)].index = Hour
        dic['Data_label_G00' + str(i_gen)].columns = Data_label_G.columns
    if 10 <= i_gen < 100:
        dic['Data_label_G0' + str(i_gen) + '_path'] = urljoin(url_root, 'Data_label_G0' + str(i_gen) + '.csv')
        dic['Data_label_G0' + str(i_gen)] = np.genfromtxt(dic['Data_label_G0' + str(i_gen) + '_path'], delimiter=',')
        dic['Data_label_G0' + str(i_gen)] = pd.DataFrame(dic['Data_label_G0' + str(i_gen)])
        dic['Data_label_G0' + str(i_gen)].index = Hour
        dic['Data_label_G0' + str(i_gen)].columns = Data_label_G.columns
    if i_gen >= 100:
        dic['Data_label_G' + str(i_gen) + '_path'] = urljoin(url_root, 'Data_label_G' + str(i_gen) + '.csv')
        dic['Data_label_G' + str(i_gen)] = np.genfromtxt(dic['Data_label_G' + str(i_gen) + '_path'], delimiter=',')
        dic['Data_label_G' + str(i_gen)] = pd.DataFrame(dic['Data_label_G' + str(i_gen)])
        dic['Data_label_G' + str(i_gen)].index = Hour
        dic['Data_label_G' + str(i_gen)].columns = Data_label_G.columns

# Get train and test data
for i_gen in Set_gen:
    if i_gen < 10:
        dic['Train_label_G00' + str(i_gen)] = dic['Data_label_G00' + str(i_gen)].loc[:,'Day_0001':'Day_3648']
        dic['Test_label_G00'  + str(i_gen)] = dic['Data_label_G00' + str(i_gen)].loc[:,'Day_3649':'Day_4384']
    if 10 <= i_gen < 100:
        dic['Train_label_G0' + str(i_gen)] = dic['Data_label_G0' + str(i_gen)].loc[:,'Day_0001':'Day_3648']
        dic['Test_label_G0'  + str(i_gen)] = dic['Data_label_G0' + str(i_gen)].loc[:,'Day_3649':'Day_4384']
    if i_gen >= 100:
        dic['Train_label_G' + str(i_gen)] = dic['Data_label_G' + str(i_gen)].loc[:,'Day_0001':'Day_3648']
        dic['Test_label_G' + str(i_gen)] = dic['Data_label_G' + str(i_gen)].loc[:, 'Day_3649':'Day_4384']

Train_feature = Data_feature.loc[:,'Day_0001':'Day_3648']
Test_feature  = Data_feature.loc[:,'Day_3649':'Day_4384']

# Do Grid Search for identifying the best parameters
param_grid = {'hidden_layer_sizes':[(150, 150, 150),
                                    (150, 100, 50),
                                    (150, 150, 150, 150, 150, 150),
                                    (150, 125, 100, 75, 50, 25)],
              'alpha': [0.000001, 0.001, 1]}
for i_gen in Set_gen:
    if i_gen < 10:
        dic['DNN_G00' + str(i_gen)] = MLPClassifier(max_iter = 50000, solver = 'adam', activation = 'relu')
        dic['DNN_GS_G00' + str(i_gen)] = GridSearchCV(dic['DNN_G00' + str(i_gen)], param_grid, cv = 10, scoring = 'accuracy')
        dic['DNN_GS_G00' + str(i_gen)].fit(Train_feature.values.T, dic['Train_label_G00' + str(i_gen)].values.T)
        dic['Best_Para_G00' + str(i_gen)] = dic['DNN_GS_G00' + str(i_gen)].best_params_
        print('Parameters for DNN_G00',str(i_gen), ':', dic['Best_Para_G00' + str(i_gen)])
    if 10 <= i_gen < 100:
        dic['DNN_G0' + str(i_gen)] = MLPClassifier(max_iter = 50000, solver = 'adam', activation = 'relu')
        dic['DNN_GS_G0' + str(i_gen)] = GridSearchCV(dic['DNN_G0' + str(i_gen)], param_grid, cv = 10, scoring = 'accuracy')
        dic['DNN_GS_G0' + str(i_gen)].fit(Train_feature.values.T, dic['Train_label_G0' + str(i_gen)].values.T)
        dic['Best_Para_G0' + str(i_gen)] = dic['DNN_GS_G0' + str(i_gen)].best_params_
        print('Parameters for DNN_G',str(i_gen), ':',  dic['Best_Para_G0' + str(i_gen)])
    if i_gen >= 100:
        dic['DNN_G' + str(i_gen)] = MLPClassifier(max_iter = 50000, solver = 'adam', activation = 'relu')
        dic['DNN_GS_G' + str(i_gen)] = GridSearchCV(dic['DNN_G' + str(i_gen)], param_grid, cv = 10, scoring = 'accuracy')
        dic['DNN_GS_G' + str(i_gen)].fit(Train_feature.values.T, dic['Train_label_G' + str(i_gen)].values.T)
        dic['Best_Para_G' + str(i_gen)] = dic['DNN_GS_G' + str(i_gen)].best_params_
        print('Parameters for DNN_G',str(i_gen), ':',  dic['Best_Para_G' + str(i_gen)])

# Use the best parameters to design DNNs
for i_gen in Set_gen:
    if i_gen < 10:
        dic['DNN_G00' + str(i_gen)] = MLPClassifier(max_iter = 50000, solver = 'adam', activation = 'relu',
                                                   hidden_layer_sizes = dic['Best_Para_G00' + str(i_gen)]['hidden_layer_sizes'],
                                                   alpha              = dic['Best_Para_G00' + str(i_gen)]['alpha'])
        dic['DNN_G00' + str(i_gen)] = dic['DNN_G00' + str(i_gen)].fit(Train_feature.T, dic['Train_label_G00' + str(i_gen)].T)
    if 10 <= i_gen < 100:
        dic['DNN_G0' + str(i_gen)] = MLPClassifier(max_iter = 50000, solver = 'adam', activation = 'relu',
                                                   hidden_layer_sizes = dic['Best_Para_G0' + str(i_gen)]['hidden_layer_sizes'],
                                                   alpha              = dic['Best_Para_G0' + str(i_gen)]['alpha'])
        dic['DNN_G0' + str(i_gen)] = dic['DNN_G0' + str(i_gen)].fit(Train_feature.T, dic['Train_label_G0' + str(i_gen)].T)
    if i_gen >= 100:
        dic['DNN_G' + str(i_gen)] = MLPClassifier(max_iter = 50000, solver = 'adam', activation = 'relu',
                                                  hidden_layer_sizes=dic['Best_Para_G' + str(i_gen)]['hidden_layer_sizes'],
                                                  alpha=dic['Best_Para_G' + str(i_gen)]['alpha'])
        dic['DNN_G' + str(i_gen)] = dic['DNN_G' + str(i_gen)].fit(Train_feature.T, dic['Train_label_G' + str(i_gen)].T)

# Check the scores
for i_gen in Set_gen:
    if i_gen < 10:
        dic['DNN_pred_G00' + str(i_gen)] = dic['DNN_G00' + str(i_gen)].predict(Test_feature.T).T
        dic['DNN_true_G00' + str(i_gen)] = np.array(dic['Test_label_G00' + str(i_gen)])
        print('-----------------------------------------------------------------------------------------------')
        print('Scores for G00', str(i_gen))
        print(classification_report(dic['DNN_true_G00' + str(i_gen)].flatten(), dic['DNN_pred_G00' + str(i_gen)].flatten()))
        if len(np.unique(dic['DNN_true_G00' + str(i_gen)])) > 1:
            print('ROC_score         ', roc_auc_score(dic['DNN_true_G00' + str(i_gen)].flatten(), dic['DNN_pred_G00' + str(i_gen)].flatten()))
        else:
            print('Only one class, no ROC score.')
            print('-----------------------------------------------------------------------------------------------')

    if 10 <= i_gen < 100:
        dic['DNN_pred_G0' + str(i_gen)] = dic['DNN_G0' + str(i_gen)].predict(Test_feature.T).T
        dic['DNN_true_G0' + str(i_gen)] = np.array(dic['Test_label_G0' + str(i_gen)])
        print('-----------------------------------------------------------------------------------------------')
        print('Scores for G0', str(i_gen))
        print(classification_report(dic['DNN_true_G0' + str(i_gen)].flatten(), dic['DNN_pred_G0' + str(i_gen)].flatten()))
        if len(np.unique(dic['DNN_true_G0' + str(i_gen)])) > 1:
            print('ROC_score         ', roc_auc_score(dic['DNN_true_G0' + str(i_gen)].flatten(), dic['DNN_pred_G0' + str(i_gen)].flatten()))
        else:
            print('Only one class, no ROC score.')
            print('-----------------------------------------------------------------------------------------------')
    if i_gen >= 100:
        dic['DNN_pred_G' + str(i_gen)] = dic['DNN_G' + str(i_gen)].predict(Test_feature.T).T
        dic['DNN_true_G' + str(i_gen)] = np.array(dic['Test_label_G' + str(i_gen)])
        print('-----------------------------------------------------------------------------------------------')
        print('Scores for G', str(i_gen))
        print(classification_report(dic['DNN_true_G' + str(i_gen)].flatten(), dic['DNN_pred_G' + str(i_gen)].flatten()))
        if len(np.unique(dic['DNN_true_G' + str(i_gen)])) > 1:
            print('ROC_score         ', roc_auc_score(dic['DNN_true_G' + str(i_gen)].flatten(), dic['DNN_pred_G' + str(i_gen)].flatten()))
        else:
            print('Only one class, no ROC score.')
            print('-----------------------------------------------------------------------------------------------')

