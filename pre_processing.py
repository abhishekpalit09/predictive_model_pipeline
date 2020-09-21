# Callable Functions
# Author: Abhishek Palit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
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
from sklearn.preprocessing import OneHotEncoder,RobustScaler,OrdinalEncoder
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
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


from statsmodels.stats.outliers_influence import variance_inflation_factor


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
        X = X.sample(frac=0.5,random_state=9999)
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
        return X
    
    
################################# Variable Selector ############################    
def Variable_Selector(Model,X,y,var_imp_file) :
     
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
          
     return X_select



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
def print_classification_performance2class_report(model,X_test,y_test):
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
    precision, recall, _ = precision_recall_curve(y_test,y_pred_proba)
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
    #tmp = pd.DataFrame(data=[y_test,y_pred_proba]).transpose()
    tmp = pd.DataFrame(data=[pd.Series(y_test).reset_index().drop(columns=['index']).transpose(),
                         pd.Series(y_pred_proba)]).transpose()
    tmp.columns=['class','proba']
    mask_c0 = tmp['class']==0
    mask_c1 = tmp['class']==1
    plt.hist(tmp.loc[mask_c0,'proba'].dropna(),density=True,alpha=0.5,label='0',bins=20)
    plt.hist(tmp.loc[mask_c1,'proba'].dropna(),density=True,alpha=0.5,label='1',bins=20)
    plt.ylabel('Density')
    plt.xlabel('Probability')
    plt.title('2-class Distribution' )
    plt.legend(loc='upper right')
    
    plt.show()
    
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
            
            # elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
            #   missing_indicator_indices = transformer.indicator_.features_
            #   missing_indicators = [raw_col_name[idx] + '_missing_flag' for idx in missing_indicator_indices]

            #  names = raw_col_name + missing_indicators
            
          else:
            names = list(transformer.get_feature_names())
          
        except AttributeError as error:
          names = raw_col_name
        
        print(names)    
        
        col_name.extend(names)
            
    return col_name



################################# Model Comparison ############################    
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)   

def model_comparison(X_train ,y):
    
    # Comparing models across K-Cross Validations
    print(datetime.now())
    random_state = 222
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(XGBClassifier(random_state=random_state))
    classifiers.append(LGBMClassifier(random_state=random_state))
    # classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(CatBoostClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    classifiers.append(LinearDiscriminantAnalysis())
    
    cv_results = []
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, X_train, y, scoring = "accuracy", cv = kfold, n_jobs=-1))
    
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
    
    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","XGBoost","LGBMBoost","RandomForest","ExtraTrees","CatBoost","mlp","knn","logistic","linear"]})
    
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    print(cv_res)
    print(datetime.now())
    
    return cv_res

################################# Hyper Parameter Tuning ############################    

    