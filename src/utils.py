import pandas as pd
import numpy as np

# fill missing value by flag feature
def fill_missing_value_by_group_neighbor(df, to_fill_feature, flag_feature_list):
    """
    Fill missing values of a columns based on the flag_feature
    """ 
  
    df[to_fill_feature] = df.sort_values(by=[to_fill_feature]).groupby(flag_feature_list)[to_fill_feature].transform(
        lambda grp: grp.fillna(method="ffill")  
    ) 

    return df


def fill_missing_value_by_neighbor(df, to_fill_feature, flag_feature_list):
    """
    Fill missing values of a columns based on the flag_feature
    """ 
  
    df[to_fill_feature] = df.sort_values(by=flag_feature_list)[to_fill_feature].transform(
        lambda grp: grp.fillna(method="ffill")  
    ) 

    return df



def fill_missing_value_by_mode(df, feature):
    """
    This method only work for categorical features
    """
    
    mode = None
    # check dtype is object
    if df[feature].dtype == object:
        mode = df[feature].value_counts().index.values[0]
        df.loc[df[feature].isnull(), feature] = mode
        
    return (df, mode)


def update_link_to_tableau(df):
    """
    Only work for my local enviroment.
    Update train.csv for virsualization.
    """
    tableau_path = './../cs5228-project/cs5228-2021-semester-2-final-project/'
    file_name = 'train.csv'
    df.to_csv(path+file_name)
    
    
def compute_null_count(df, feature_list=None):
    
    if not feature_list:
        return np.sum(pd.isnull(df))
    else:
        return np.sum(pd.isnull(df[feature_list]))