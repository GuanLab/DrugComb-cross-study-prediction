import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

def train_RandomForest(X, Y):
    """ Train Random Forest models
    """
    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X, Y)
    return regressor

def train_LightGBM(X,Y, feature_names):#, concen_only=False):
    """ Train LightGBM models
    
    Paramteres:
    -----------
    X: Numpy array
    Y: Numpy array
    feature_name: list of strings
    
    Yields:
    -------
    regressor: the light GBM model
    """
    param = {'boosting_type': 'gbdt',
            'objective': 'regression', #'quantile'
            'num_leaves': 20,
            'max_depth': 8,
            'force_col_wise': 'true',
            #'min_data_in_leaf': 2000,
            'learning_rate': 0.05,
            'verbose': 0,
            'n_estimators': 1000,
            'reg_alpha': 2.0,
                   }

    categorical_feature = ["cell_line_categorical", "drug_categorical_row", "drug_categorical_col"]
    categorical_feature = [x for x in categorical_feature if x in feature_names]
    
    # model training 
    
    """
    if concen_only:
        train_data = lgb.Dataset(data = X,
                                 label = Y,
                                 feature_name = feature_names)
    else:
    """
    train_data = lgb.Dataset(data = X,
                            label = Y,
                            feature_name = feature_names,
                            categorical_feature = categorical_feature)
    
    #val_data = lgb.Dataset(data = val_X,
    #        label = val_Y,
    #        feature_name = feature_name,
    #        categorical_feature = categorical_feature)

    regressor = lgb.train(param, 
            train_set = train_data,
            num_boost_round= 500)

    
    return regressor
