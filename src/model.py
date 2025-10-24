import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def encode_categorical_features(df):
    """
    Encode categorical features to numerical values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing categorical features
        
    Returns:
    --------
    pandas.DataFrame
        Encoded DataFrame
    """
    df = df.copy()
    # 性别编码
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    return df


def prepare_training_data(train_df, predictors, target, test_size=0.2, random_state=42):
    """
    Prepare training data including feature and label separation and train-validation split
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame
    predictors : list
        List of predictor feature column names
    target : str
        Target variable column name
    test_size : float
        Validation set proportion
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (train_X, train_Y, valid_X, valid_Y) Training and validation features and labels
    """
    train, valid = train_test_split(train_df, test_size=test_size, random_state=random_state, shuffle=True)
    
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    
    return train_X, train_Y, valid_X, valid_Y


def train_random_forest(train_X, train_Y, n_estimators=100, random_state=42, criterion="gini", n_jobs=-1):
    """
    Train a Random Forest classifier
    
    Parameters:
    -----------
    train_X : pandas.DataFrame or numpy.ndarray
        Training features
    train_Y : numpy.ndarray
        Training labels
    n_estimators : int
        Number of decision trees
    random_state : int
        Random seed
    criterion : str
        Splitting criterion
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained Random Forest classifier
    """
    clf = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
        criterion=criterion,
        n_estimators=n_estimators,
        verbose=False
    )
    clf.fit(train_X, train_Y)
    return clf


def evaluate_model(model, X, y, target_names=None):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classifier
    X : pandas.DataFrame or numpy.ndarray
        Feature data
    y : numpy.ndarray
        True labels
    target_names : list, optional
        Target class names
        
    Returns:
    --------
    dict
        Dictionary containing predictions and classification report
    """
    predictions = model.predict(X)
    
    if target_names is None:
        target_names = ['Not Survived', 'Survived']
    
    classification_report = metrics.classification_report(y, predictions, target_names=target_names)
    
    return {
        'predictions': predictions,
        'classification_report': classification_report
    }


def get_model_predictions(model, X):
    """
    Get model predictions
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classifier
    X : pandas.DataFrame or numpy.ndarray
        Feature data
        
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    return model.predict(X)
