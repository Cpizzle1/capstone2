from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, accuracy_score, f1_score,accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVC

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import recall_score, precision_score, roc_curve, auc, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import defaultdict
# from roc import plot_roc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier




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
    # y_train.ravel()
    y_train = np.array(y_train).reshape(-1)

    logistic2_regression_grid = {'C':[0.0305,0.03055, 0.03060, 0.03065, 0.0307, 0.03075, 0.03077]
#                        ,'cv':[4]
                       ,'solver':['liblinear']#'lbfgs',
#                        ,'max_iter' : [50]
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

    print(y_train.shape)

