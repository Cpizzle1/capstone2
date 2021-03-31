from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import  roc_auc_score, roc_curve, auc,  accuracy_score,f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import pickle

# from roc import plot_roc

import matplotlib.pyplot as plt


from sklearn.svm import SVC





def optimize_model2_randomCV(model, grid_params, X_train, y_train, scoring):

    model_search = RandomizedSearchCV(model
                                        ,grid_params
                                        ,n_jobs=-1
                                        ,verbose=False
                                        ,scoring=scoring)
    model_search.fit(X_train, y_train)
    print(f"Best Parameters for {model}: {model_search.best_params_}")
    print(f"Best Model for {model}: {model_search.best_estimator_}")
    print(f"Best Score for {model}: {model_search.best_score_:.4f}")
    
    return model_search.best_estimator_







if __name__ == '__main__':
  

    X_train = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/rna_jupyternotebook_df_wednesday.csv')
    y_train = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/target_death_short_col.csv')
 
    y_train = np.array(y_train).reshape(-1)

    logistic2_regression_grid = {'C':[0.0305,0.03055, 0.03060, 0.03065, 0.0307, 0.03075, 0.03077]
#                        ,'cv':[4]
                       ,'solver':['liblinear']#'lbfgs',

                       ,'class_weight':['balanced']
                       ,'penalty':['l1']} #, 'l2', 'elasticnet'

    logistic_regressionCV_grid = {'Cs':[2,5,10, 25, 100, 200]
                       ,'cv':[4]
                       ,'solver':['liblinear']#'lbfgs',
#                        ,'max_iter' : [50]
                       ,'class_weight':['balanced']
                       ,'penalty':['l1'] #, 'l2', 'elasticnet'
                        }

    random_forest_grid = {'max_depth': [2, 4, 8]
                     ,'max_features': ['sqrt', 'log2', None]
                     ,'min_samples_leaf': [1, 2, 4]
                     ,'min_samples_split': [2, 4]
                     ,'bootstrap': [True, False]
                     ,'class_weight': ['balanced']

                     ,'n_estimators': [5,10,25,50,100,200]
                     }
                    
    gradient_boosting_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
                         ,'max_depth': [2, 4, 8]
                         ,'subsample': [0.25, 0.5, 0.75, 1.0]
                         ,'min_samples_leaf': [1, 2, 4]
                         ,'max_features': ['sqrt', 'log2', None]
                         ,'n_estimators': [5,10,25,50,100,200]
                         }

    
    

    
    
    # logistic2_randomsearch = RandomizedSearchCV(LogisticRegression()
    #                                           ,logistic2_regression_grid
    #                                           ,n_jobs=-1
    #                                           ,verbose=False
    #                                           ,scoring='roc_auc')


    results = optimize_model2_randomCV(LogisticRegression(), logistic2_regression_grid, X_train, y_train, scoring= 'roc_auc')




