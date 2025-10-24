import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_count_pairs(df, column, hue=None, title=None, figsize=(8, 4), color_list=None):
    """
    Plot count pairs
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame
    column : str
        Column name to plot
    hue : str, optional
        Column name for grouping
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    color_list : list, optional
        Color list
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    
    if hue:
        sns.countplot(x=column, data=df, hue=hue, palette=color_list)
    else:
        sns.countplot(x=column, data=df, palette=color_list)
    
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    if title:
        ax.set_title(title)
    plt.show()


def plot_distribution_pairs(df, column, hue=None, title=None, figsize=(8, 4), color_list=None):
    """
    Plot distribution pairs
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame
    column : str
        Column name to plot
    hue : str, optional
        Column name for grouping
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    color_list : list, optional
        Color list
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    
    if hue:
        for i, h in enumerate(df[hue].unique()):
            g = sns.histplot(df.loc[df[hue] == h, column], 
                           color=color_list[i] if color_list else None, 
                           ax=ax, 
                           label=h)
        g.legend()
    else:
        sns.histplot(df[column], ax=ax)
    
    if title:
        ax.set_title(title)
    plt.show()


def print_data_summary(df, dataset_name="Dataset"):
    """
    Print basic information summary of a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame
    dataset_name : str
        Dataset name
    """
    print(f"\n=== {dataset_name} Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def get_feature_importance(model, feature_names):
    """
    Get feature importance
    
    Parameters:
    -----------
    model : sklearn ensemble model
        Trained model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance
    top_n : int
        Number of top important features to display
    figsize : tuple
        Figure size
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
