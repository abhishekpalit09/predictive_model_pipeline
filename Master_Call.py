# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:26:10 2020

@author: RSAGEAS1
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:45:00 2020

@author: RSAGEAS1
"""

import os
# Working Directory
os. chdir("C:/Users/RSAGEAS1/Documents/Abhi") 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pyodbc
import sys
import warnings
from pre_processing import *
# %matplotlib inline
print(datetime.now())

random.seed(9999)
# Sampling for model comparision
sampling_mdl_cmp = 1

# Sampling for var selection
sampling_var_select = 1

# Study trial
n_trial=20

# Oversampled Value
Over_Sample_perc = .1

# Comparison Algo
algo = CatBoostClassifier(iterations = 100,random_state=222)

######################## Fetching from db ###################################
# Using a DSN
conn = pyodbc.connect('DSN=RSGI_ageas;UID=ageas;PWD=ageas@2019')

query = "SELECT * FROM RSA_IRCTC.dbo.IRCTC_FINAL WHERE statename in ('MAHARASTRA','DELHI','UTTAR PRADESH','KARNATAKA','TAMIL NADU')"
data =pd.read_sql(query,conn)


######################## Data Processing   ###################################
data.drop(columns = ['EMAIL','MAX_AGE',	'MAX_FAMILY_RANK',	'FAMILY_RANK',	'AGE',	'PINCODE',	'COACH_NAME',	'FAMILY_TYPE',	'PROD_PA',	'Districtname',	'statename'],axis = 1,inplace = True)


# Feature Engineering
data['WEND_FLAG']=np.where(data['SUM_WEND_FLAG']>0,1,0)
data['TRAVEL_Q1_FLAG']=np.where(data['TRAVEL_Q1']>0,1,0)
data['TRAVEL_Q2_FLAG']=np.where(data['TRAVEL_Q2']>0,1,0)
data['TRAVEL_Q3_FLAG']=np.where(data['TRAVEL_Q3']>0,1,0)
data['TRAVEL_Q4_FLAG']=np.where(data['TRAVEL_Q4']>0,1,0)


# Cat_Ordinal = ['Saving_accounts','Checking_account']
# regionname SUM_TRVL_BKNG_DIFF dropped manually
Cont = ['TOT_TRIP','SUM_TRVL_BKNG_DIFF',	'SELF_TRAVEL',	'MOB_COUNT',	'SUM_DURATION',	'AVG_DURATION',	'SUM_WEND_FLAG',	'AVG_TRVL_BKNG_DIFF',	'MAX_TICKET_SIZE',	'AVG_TICKET_SIZE',	'TRAVEL_Q1',	'TRAVEL_Q2',	'TRAVEL_Q3',	'TRAVEL_Q4'
        ,'WEND_FLAG','TRAVEL_Q1_FLAG','TRAVEL_Q2_FLAG','TRAVEL_Q3_FLAG','TRAVEL_Q4_FLAG']

# regionname dropped manually
Cat_Onehot = ['SEX',	'FREQ_P_AGE',	'DEP_FAMILY_TYPE',	'COACH_TYPE']

# Target = ['PROD_CAR']

Vehicle = pd.DataFrame(np.where(data['PROD_CAR']>0,1,0),columns=['Vehicle'])
Mbike = pd.DataFrame(np.where(data['PROD_MBIKE']>0,1,0),columns=['Mbike'])
Health_Premium = pd.DataFrame(np.where((data['PROD_HEALTH_EL']>0)|(data['PROD_HEALTH_SP']>0),1,0),columns=['Health_Premium'])
Health_Normal = pd.DataFrame(np.where((data['PROD_HEALTH_AR']>0)|(data['PROD_HEALTH_CL']>0),1,0),columns=['Health_Normal'])


####################### Multi Colinearity Treatment #######################
# Correlation Matrix
corr_mat = data[Cont].corr()
corr_mat.to_csv('Corr_matrix.csv')
sns.heatmap(corr_mat,annot = True)
plt.show()

#One Time Activity
# Cont_Revised=ReduceVIF().fit_transform(pd.DataFrame(data,columns = Cont))
# Cont_Revised = df_Cont.columns
Cont_Revised = ['WEND_FLAG',	'SELF_TRAVEL',	'MOB_COUNT',	'AVG_DURATION',	'AVG_TRVL_BKNG_DIFF',	'AVG_TICKET_SIZE',	'TRAVEL_Q1_FLAG',	'TRAVEL_Q2_FLAG',	'TRAVEL_Q3_FLAG',	'TRAVEL_Q4_FLAG']

corr_mat_revised = data[Cont_Revised].corr()
corr_mat_revised.to_csv('Corr_matrix_revised.csv')
####################### Data Pipeline #######################
# Data Pipeline
df_processed = data_pipeline(Cont_Revised,Cat_Onehot,data)

model_data_colnames = list(df_processed.columns)

with open('model_data_colnames','wb') as fp:
    pickle.dump(model_data_colnames,fp)

del data

####################### Final Model #######################
for model_name in [Vehicle,Mbike,Health_Premium,Health_Normal]:
# for model_name in [Vehicle]:    
    var_imp_file = './'+model_name.columns[0]+'/top5_var_imp_file.csv'
    mdl_stat_file = './'+model_name.columns[0]+'/top5_mdl_stat_file.csv'
    sampling_stats_file = './'+model_name.columns[0]+'/top5_sampling_stats_file.csv'
    study_param_file = './'+model_name.columns[0]+'/top5_study_param_file.txt'
    
    img_file = './'+model_name.columns[0]+'/model_perf.png'
    mdl_file = './'+model_name.columns[0]+'/'+model_name.columns[0]+'_model.pkl'
    var_select_file = './'+model_name.columns[0]+'/'+model_name.columns[0]+'_var_select'
    
    
    feat_labels =Var_Selection(Model =algo,X = df_processed,y =model_name
                               ,var_imp_file = var_imp_file
                               ,sampling = sampling_var_select
                               # ,var_select = var_select_file
                               )
    
    with open(var_select_file,'wb') as fp:
        pickle.dump(feat_labels,fp)
    
    model_data = df_processed[feat_labels]
    
    sampler = SMOTE(sampling_strategy=Over_Sample_perc)
    
    # Splitting 70/30 game
    X_train,X_test,y_train,y_test = train_test_split(model_data,model_name, test_size =.3, stratify = Vehicle)
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_train, y_train = sampler.fit_resample(X_train,y_train)
    print(Counter(y_train))
     
    objective(data=model_data,y=model_name,oversampling=oversampling_choice,
             sampler=sampler,classifier=algo_name,sampling=sampling_mdl_cmp,
             file=study_param_file,ntrial=n_trial)


    params = {
        # 'classifier': 'LGBMClassifier',
              'lambda_l1': 0.03130906392696624,
              'lambda_l2': 4.466190413472027e-07,
              'num_leaves': 159,
              'feature_fraction': 0.7725116411866746, 
              'bagging_fraction': 0.600280119352165, 
              'bagging_freq': 3, 'min_child_samples': 29}

    lgbm = LGBMClassifier(**params)
    # lgbm = CatBoostClassifier(iterations = 500,random_state=222)
    # lgbm = LGBMClassifier(random_state=222)
    best_model = lgbm.fit(X_train,y_train)
    
    print_classification_performance2class_report(best_model,X_test,y_test,file_name = img_file)
    # plot_precision_recall_vs_threshold(precisions=PC, recalls=RC, thresholds=.05)
    
    model_performance_on_threshold(best_model,X_test,y_test,threshold=.05)
    
    joblib.dump(best_model,mdl_file)
    
    
      
print(datetime.now())
