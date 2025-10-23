
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame


def make_split(
    df: pd.DataFrame,
    valid_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True
) -> Split:
    """Create a train/validation split from the labelled training dataframe."""
    train, valid = train_test_split(
        df, test_size=valid_size, random_state=random_state, shuffle=shuffle
    )
    return Split(train=train, valid=valid)


def get_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    predictors: Sequence[str],
    target: str
):
    """Return X_train, y_train, X_valid, y_valid according to predictors/target."""
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    return train_X, train_Y, valid_X, valid_Y


def fit_predict_rf(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    criterion: str = "gini",
    verbose: bool = False
):
    """
    Fit a RandomForestClassifier and return fitted model and validation predictions.

    Returns
    -------
    model : RandomForestClassifier
    preds : np.ndarray
    metrics_dict : dict with 'classification_report' (as dict) and 'auc' (float)
    """
    Xtr, ytr, Xva, yva = get_xy(train_df, valid_df, predictors, target)
    clf = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
        criterion=criterion,
        n_estimators=n_estimators,
        verbose=verbose
    )
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xva)
    proba = clf.predict_proba(Xva)[:, 1] if hasattr(clf, "predict_proba") else None
    report = metrics.classification_report(
        yva, preds, target_names=['Not Survived', 'Survived'], output_dict=True
    )
    auc = metrics.roc_auc_score(yva, proba) if proba is not None else None
    return clf, preds, {'classification_report': report, 'auc': auc}
