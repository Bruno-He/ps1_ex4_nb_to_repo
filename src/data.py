

from __future__ import annotations
from pathlib import Path
import pandas as pd

DEFAULT_TRAIN = Path("C:/Users/Bronny/PS1_EX4_NB_TO_REPO/data/train.csv")
DEFAULT_TEST = Path("C:/Users/Bronny/PS1_EX4_NB_TO_REPO/data/test.csv")


def load_data(
    train_path: str | Path = DEFAULT_TRAIN,
    test_path: str | Path = DEFAULT_TEST
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:

    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.shape[0] * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = [str(df[col].dtype) for col in df.columns]
    tt['Types'] = types
    uniques = [df[col].nunique(dropna=True) for col in df.columns]
    tt['Uniques'] = uniques
    return tt.T

def most_frequent_summary(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    total = df.count()
    tt = pd.DataFrame(total, columns=['Total'])
    items = []
    vals = []
    for col in df.columns:
        try:
            vc = df[col].value_counts()
            itm = vc.index[0]
            val = vc.values[0]
            items.append(itm)
            vals.append(val)
        except Exception:
            items.append(None)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return tt.T


def nonnull_unique_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = df.count()
    tt = pd.DataFrame(total, columns=['Total'])
    uniques = [df[col].nunique(dropna=True) for col in df.columns]
    tt['Uniques'] = uniques
    return tt.T


def concat_with_set(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df['set'] = 'train'
    if 'Survived' in all_df.columns:
        all_df.loc[all_df['Survived'].isna(), 'set'] = 'test'
    return all_df


def inspect_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame, head_n: int = 5) -> None:
    print("\n===== TRAIN SET =====")
    display(train_df.head(head_n))
    print("\nShape:", train_df.shape)
    print("\nInfo:")
    display(train_df.info())

    print("\n===== TEST SET =====")
    display(test_df.head(head_n))
    print("\nShape:", test_df.shape)
    print("\nInfo:")
    display(test_df.info())
