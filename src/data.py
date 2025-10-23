import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def analyze_missing_data(df):
    """
    Analyze missing data in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Transposed DataFrame containing missing data statistics including:
        - Total: Total number of missing values
        - Percent: Percentage of missing values
        - Types: Data types
    """
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    
    return np.transpose(tt)


def analyze_frequent_data(df):
    """
    Analyze the most frequent values and frequencies for each column in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Transposed DataFrame containing most frequent data statistics including:
        - Total: Total number of non-null values
        - Most frequent item: Most frequent value
        - Frequence: Count of most frequent value
        - Percent from total: Percentage of most frequent value from total
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    items = []
    vals = []
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(f"Error processing column {col}: {ex}")
            items.append(0)
            vals.append(0)
            continue
    
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    
    return np.transpose(tt)


def analyze_unique_values(df):
    """
    Analyze the number of unique values for each column in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Transposed DataFrame containing unique values statistics including:
        - Total: Total number of non-null values
        - Uniques: Number of unique values
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    
    return np.transpose(tt)

