# Callable Functions
# Author: Abhishek Palit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import optuna
import pickle
# from sklearn.externals 
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer  as Imputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix , average_precision_score, accuracy_score,silhouette_score
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,RobustScaler,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine import outlier_removers as outr
from sklearn.impute import SimpleImputer
from inspect import signature
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN,SMOTENC,KMeansSMOTE
from sklearn.model_selection import cross_validate
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functools import partial

# k-fold datasets
sets = 3

#################################  Abhi_Trasformer   ############################
class Abhi_Trasformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self



################################# Outlier Treatment ############################
class NewVariablesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
    # Make a new variable that is rating divided by number of reviews
        ratings_over_reviews = X[:,ratings_index]/X[:,reviews_index]
        return np.c_[X, ratings_over_reviews]


def outlier_treatment(df):
       
   for columns in df:
    
        Q1 = df[columns].quantile(0.25) 
        Q2 =df[columns].quantile(0.5)   
        Q3 = df[columns].quantile(0.75)
        
#         # Interquartile range (IQR)
        IQR = Q3 - Q1
        
#         # outlier step
        outlier_step = 1.5 * IQR
    
        df.loc[ :,columns] = np.where(df.loc[ :,columns]>(Q3 + outlier_step),Q2,df.loc[ :,columns])  
        
   return df     


#################################  Missing Variable Treatment ############################
def missingvar_treatment(df):
    
    # Missing Variables
    df = df.fillna(df.mean())

    df.isnull().sum()
    
    return df


#################################  Variable Standardization ############################
def var_std(df):
    
   # RobustScaler is less prone to outliers.
   rob_scaler = RobustScaler()

   for columns in df:
    
    df[columns] = rob_scaler.fit_transform(df[columns].values.reshape(-1,1))
   
   return df


################################# Categorical Variable Treatment: Label Encoding ############################
def Label_Encoding(df):
    
    # Convert to categorical Variable
   for columns in df:
        df.loc[ :,columns] = df.loc[ :,columns].astype('category')
    
    
    # Label encoding
   for columns in df:
        df.loc[ :,columns]  =df.loc[ :,columns].cat.codes
       
   return df



################################# Categorical Variable Treatment: One Hot Encoding ############################

def OneHot_Encoding(df):
        
    # Convert to categorical Variable
   for columns in df:
        df.loc[ :,columns] = df.loc[ :,columns].astype('category')
        
    # Label encoding
   df = pd.get_dummies(df)
        
   df.head()
    
   return df

################################# Multi-Colinearity Treatment ############################
class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        self.thresh = thresh
        
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)
    
    def get_feature_names(self):
        return self.X.values.tolist()        

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        X = X.sample(frac=0.5)
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                # drop_list.append()
                dropped=True
            # print(drop_list)
        vif_names = X.columns
        return vif_names
    
    
################################# Variable Selector ############################    
def Variable_Selector(Model,X,y,var_imp_file,sampling) :
     
     print(datetime.now())
     X = X.sample(frac=sampling)
     y = y.sample(frac=sampling)
     
     select = SelectFromModel(estimator = Model)
     select.fit(X,y)
     
     feat_labels =  X.columns

     for i in select.get_support(indices =True):
      features = feat_labels[i]
      print(features)
     
     #Significant Variables
     feat_indx = select.get_support()
     feat_names = X.columns[feat_indx]
     
     X_select = X[feat_names]  
      
     # Variale Importance Report 
     r = Model.fit(X,y)   
     with open(var_imp_file, 'w',newline ='') as file:
         writer = csv.writer(file)
         for i in zip(feat_labels,r.feature_importances_):
             var_imp = [i]
             writer.writerow(var_imp)        
     print(datetime.now())     
     return X_select


################################# Variable Selector ############################    
def Var_Selection(Model,X,y,var_imp_file,sampling) :
     
    print(datetime.now())
    X = X.sample(frac=sampling)
    yt = y.sample(frac=sampling)
    
    select = SelectFromModel(estimator = Model)
    select.fit(X,yt)
    
    #Significant Variables
    feat_labels =  X.columns[select.get_support(indices =True)]
    print(feat_labels)
      
    # Variale Importance Report 
    r = Model.fit(X,yt)   
    pd.DataFrame(zip(X.columns,r.feature_importances_)).to_csv(var_imp_file,index=False)
       
    print(datetime.now())     
    return feat_labels


################################# Survival Variable Selector ############################
def Surv_Var_Selector(Model,df,X,y):
    
    feat_labels =  X.columns
    select = SelectFromModel(estimator = Model)

    select.fit(X,y)

    for i in select.get_support(indices =True):
        features = feat_labels[i]
        print(features)
     
    feat_indx = select.get_support()
    feat_names = X.columns[feat_indx]

    X_select = X[feat_names]
    Y = df[['TARGET','TENURE']]

    model_data = pd.concat([Y,X_select], axis = 1, ignore_index =False)
    
    return model_data
   
    
 ################################# Classification Performance ############################   

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

def model_performance(model,X_test,y_test):
    """            
        Purpose: print standard 2-class classification metrics report
    """
    sns.set()
    start = datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.now()
    timed = (end-start).total_seconds() * 1000.0
    y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    PC =  TP/(TP+FP)
    RC = TP/(TP+FN)
    FS = 2 *((PC*RC)/(PC+RC))
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    
    return

def model_performance_on_threshold(best_model,X_test,y_test,threshold):

    probs = best_model.predict_proba(X_test)[:,1]
    y_test['y_prob'] = probs
    y_test['y_binary']=np.where(y_test['y_prob']>threshold,1,0)
    
    conf_mat = confusion_matrix(y_test.iloc[:,0],y_test['y_binary'])
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    PC =  TP/(TP+FP)
    RC = TP/(TP+FN)
    FS = 2 *((PC*RC)/(PC+RC))
    AP = average_precision_score(y_test.iloc[:,0],y_test['y_binary'])
    ACC = accuracy_score(y_test.iloc[:,0],y_test['y_binary'])
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    
    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)


def print_classification_performance2class_report(model,X_test,y_test,file_name):
    """ 
        Program: print_classification_performance2class_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report
    """
    sns.set()
    start = datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.now()
    timed = (end-start).total_seconds() * 1000.0
    y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    PC =  TP/(TP+FP)
    RC = TP/(TP+FN)
    FS = 2 *((PC*RC)/(PC+RC))
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    
    fig = plt.figure(figsize=(20,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    
    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)
    
    #ROC
    plt.subplot(142)
    pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    roc_auc = auc(pfr, tpr)
    gini = (roc_auc*2)-1
    plt.plot(pfr, tpr, label='ROC Curve (area =  {:.2%})'.format(roc_auc) )
    plt.plot([0,1], [0,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Charecteristic Curve with Gini {:.2}'.format(gini))
    plt.legend(loc='lower right')
    
    #pr
    plt.subplot(143)
    precision, recall,thresholds = precision_recall_curve(y_test,y_pred_proba)
    step_kwargs = ({'step':'post'}
                  if 'step'in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2, where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('2-class Precision-Recall Curve: AP={:.2%}'.format(AP))
    
    #hist
    plt.subplot(144)
    plt.plot(thresholds, precision[:-1], 'b--', label='precision')
    plt.plot(thresholds, recall[:-1], 'g--', label = 'recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    
    
    plt.show()
    plt.savefig(file_name)
    
    return ACC,PC,RC,FS,AP,roc_auc,gini,y_pred_proba,timed

 ################################# Improving Cat Boost Model ############################   
# Improving Cat boost for Voting application
class CatBoostClassifierInt(CatBoostClassifier):
    def fit(self, X, y=None, cat_features=None, text_features=None, pairs=None, sample_weight=None, group_id=None,
            group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None):
        # Handle different types of label
        self.le_ = LabelEncoder().fit(y)
        transformed_y = self.le_.transform(y)

        self._fit(X, transformed_y, cat_features, text_features, pairs, sample_weight, group_id, group_weight, subgroup_id,
                         pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot,
                         column_description, verbose_eval, metric_period, silent, early_stopping_rounds,
                         save_snapshot, snapshot_file, snapshot_interval, init_model)
        return self
        
    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=1, verbose=None):
        predictions = self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose,'predict')

        # Return same type as input
        return self.le_.inverse_transform(predictions.astype(np.int64))
    
    
################################# get_column_names_from_ColumnTransformer ############################    

def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []

    for transformer_in_columns in column_transformer.transformers_[:-1]: #the last transformer is ColumnTransformer's 'remainder'
        print('\n\ntransformer: ', transformer_in_columns[0])
        
        raw_col_name = list(transformer_in_columns[2])
        
        if isinstance(transformer_in_columns[1], Pipeline): 
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
            
        try:
          if isinstance(transformer, OneHotEncoder):
            names = list(transformer.get_feature_names(raw_col_name))
            
          elif isinstance(transformer, RobustScaler):
            names = list(transformer.get_feature_names(raw_col_name))  
            
          elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [raw_col_name[idx] + '_missing_flag' for idx in missing_indicator_indices]

            names = raw_col_name + missing_indicators
            
          else:
            names = list(transformer.get_feature_names())
          
        except AttributeError as error:
          names = raw_col_name
        
        print(names)    
        
        col_name.extend(names)
            
    return col_name



################################# Model Comparison ############################    
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=sets)   

def model_comparison(data ,y,sampling):
    
    data = data.sample(frac=sampling)
    y = y.sample(frac=sampling)
    
    # Splitting 70/30 game
    X_train,X_test,y_train,y_test = train_test_split(data,y, test_size =.3, stratify = y)

    
    # Comparing models across K-Cross Validations
    print(datetime.now())
    random_state = 222
    classifiers = []
    # classifiers.append(SVC(random_state=random_state))
    classifiers.append(XGBClassifier(random_state=random_state))
    classifiers.append(LGBMClassifier(random_state=random_state))
    # classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    # classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(CatBoostClassifier(random_state=random_state,iterations = 100))
    # classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    # classifiers.append(LinearDiscriminantAnalysis())
    
    cv_results = []
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, X_train, y_train, scoring = "accuracy", cv = kfold, n_jobs=-1))
    
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
    
    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["XGBClassifier","LGBMClassifier","RandomForestClassifier","CatBoostClassifier","KNeighborsClassifier","LogisticRegression"]})
    
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    print(cv_res)
    print(datetime.now())
    
    cv_res.sort_values(by=['CrossValMeans'],inplace=True,ascending=False)
    
    return cv_res

################################# Data Pipeline ############################    

def data_pipeline(Cont_Revised,Cat_Onehot,data):
        # Continous Variable Pipeline
        Cont_pipeline = Pipeline(steps=[
            ('outlier',outr.Winsorizer(distribution='gaussian',tail='both', fold=1)),
            ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
            # ('add_variables', NewVariablesAdder()),
            # ('outlier',Abhi_Trasformer(outlier_treatment)),
            ('scl', StandardScaler())
            ])

        # Categorical Variable Pipeline
        Cat_Ordinal_pipeline = Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
            ('Ordinal', OrdinalEncoder())])
        
        Cat_Onehot_pipeline = Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
            ('Onehot', OneHotEncoder())])
        
        dat_pipe = ColumnTransformer([
            ('cont', Cont_pipeline, Cont_Revised),
            # ('cat_ordinal', Cat_Ordinal_pipeline, Cat_Ordinal),
            ('cat_onehot', Cat_Onehot_pipeline, Cat_Onehot),
            # ('feature_selection', SelectFromModel(rfc)),
            ])
        data_pipe = dat_pipe.fit(data)
        joblib.dump(data_pipe,'data_pipeline.pkl')
        
        df_processed =pd.DataFrame.sparse.from_spmatrix(dat_pipe.fit_transform(data), columns=get_column_names_from_ColumnTransformer(dat_pipe))
        
        return df_processed
      
################################# Best Sampling Method ############################    

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=sets)   

def sampling_comparison(data ,y,minority_sample,classifier,sampling):
    
    data = data.sample(frac=sampling)
    y = y.sample(frac=sampling)
    
    scoring = {
    'acc' : 'accuracy',
    'prec_macro' : 'precision_macro',
    'rec_macro' : 'recall_macro'
    }
    
    print("Starting Sampling Engine")
    print(datetime.now())
    
    X_train,X_test,y_train,y_test = train_test_split(data,y, test_size =.3, stratify = y)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    samplers = []
    cv_results = []
    cv_acc = []
    cv_precision = []
    cv_recall = []
    
    # Comparing models across K-Cross Validations
    
    Over_20 = 2*minority_sample
    # Over_30 = 3*minority_sample
    Over_50 = 5*minority_sample
        
    samplers.append(SMOTE(sampling_strategy=minority_sample))
    samplers.append(BorderlineSMOTE(sampling_strategy=minority_sample))
    samplers.append(ADASYN(sampling_strategy=minority_sample))
    
    samplers.append(SMOTE(sampling_strategy=Over_20))
    samplers.append(BorderlineSMOTE(sampling_strategy=Over_20))
    # samplers.append(SVMSMOTE(sampling_strategy=minority_sample))
    samplers.append(ADASYN(sampling_strategy=Over_20))
    
    samplers.append(SMOTE(sampling_strategy=Over_50))
    samplers.append(BorderlineSMOTE(sampling_strategy=Over_50))
    # samplers.append(SVMSMOTE(sampling_strategy=Over_50))
    samplers.append(ADASYN(sampling_strategy=Over_50))
    
    
    for sampler in samplers :
        
        X_SMOTE, y_SMOTE = sampler.fit_resample(X_train,y_train)
        cv_results.append(cross_validate(classifier, X_SMOTE, y_SMOTE, scoring = scoring, cv = kfold, n_jobs=-1))
        print(Counter(y_SMOTE))
        model = classifier.fit(X_SMOTE,y_SMOTE)
        model_performance(classifier,X_test,y_test)
        X_SMOTE = None
        y_SMOTE = None
        model = None
    
    for cv_result in cv_results:
        cv_acc.append(cv_result.get('test_acc').mean())
        cv_precision.append(cv_result.get('test_prec_macro').mean())
        cv_recall.append(cv_result.get('test_prec_macro').mean())
    
    # cv_sampling = pd.DataFrame({"CrossValAcc":cv_acc,"CrossValprecision": cv_precision,"CrossValRecall":cv_recall,"Sampler":[ "SMOTE","BorderlineSMOTE","SVMSMOTE","ADASYN"]})
    # cv_sampling = pd.DataFrame({"CrossValAcc":cv_acc,"CrossValprecision": cv_precision,"CrossValRecall":cv_recall,"Sampler":[ "SMOTE","BorderlineSMOTE","ADASYN"]})
    cv_sampling = pd.DataFrame({"CrossValAcc":cv_acc,"CrossValprecision": cv_precision,"CrossValRecall":cv_recall,"Sampler":samplers})
    print(datetime.now())

    cv_sampling.sort_values(by=['CrossValprecision'],inplace=True,ascending=False)
    
    
    return cv_sampling



################################# Hyper Parameter Tuning ############################    

def optimize(trial,classifier,X_train, y_train):

    # classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest","LGBMClassifier","CatBoostClassifier"])
    classifier_name = trial.suggest_categorical("classifier", classifier)
    # classifier_name = classifier
    # classifier_obj = []
    
    # Step 2. Setup values for the hyperparameters:
    if classifier_name == 'LogReg':
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        classifier_obj = LogisticRegression(C=logreg_c)
    elif classifier_name == 'RandomForest':
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 500)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators
        )
    elif classifier_name == 'LGBMClassifier':
        param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
    
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        
        classifier_obj = LGBMClassifier(
            **param, callbacks=[pruning_callback]
        )
    elif classifier_name == 'CatBoostClassifier':
        param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        }
    
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        
        classifier_obj = CatBoostClassifier(**param)
    
    elif classifier_name == 'XGBClassifier':
        param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }
    
        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
        classifier_obj = XGBClassifier(**param)
        
        
    # Step 3: Scoring method:
    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

#Step 1. Define an objective function to be maximized.
def objective(data,y,oversampling,sampler,classifier,sampling,file,n_trial):
    
    data = data.sample(frac=sampling)
    y = y.sample(frac=sampling)
    
    # Splitting 70/30 game
    X_train,X_test,y_train,y_test = train_test_split(data,y, test_size =.3, stratify = y)
    
    if oversampling== True:
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_train, y_train = sampler.fit_resample(X_train,y_train)
        print(Counter(y_train))
        
    optimize_function = partial(optimize,classifier=classifier
                                ,X_train=X_train, y_train=y_train)
        
    
    # Finding the Optimal Model
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_function, n_trials=n_trial)
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    
    print("  Params: ")
    f =open(file,'w')
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value),file = f)
    f.close()

   
