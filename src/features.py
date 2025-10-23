
from __future__ import annotations
import pandas as pd
import numpy as np


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with 'Family Size' = SibSp + Parch + 1."""
    out = df.copy()
    out["Family Size"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    return out


def bin_continuous(col: pd.Series, bins, labels=None) -> pd.Series:
    """Generic binning helper based on pandas.cut."""
    return pd.cut(col, bins=bins, labels=labels, include_lowest=True, right=True)


def add_age_interval(
    df: pd.DataFrame,
    bins=(-np.inf, 16, 32, 48, 64, np.inf),
    labels=(0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Add 'Age Interval' categorical/bin column using fixed cut points."""
    out = df.copy()
    out['Age Interval'] = pd.cut(
        out['Age'], bins=bins, labels=labels, include_lowest=True, right=True
    )
    out['Age Interval'] = out['Age Interval'].astype(float)
    return out


def add_fare_interval(
    df: pd.DataFrame,
    cuts=(-np.inf, 7.91, 14.454, 31, np.inf),
    labels=(0, 1, 2, 3),
) -> pd.DataFrame:
    """Add 'Fare Interval' categorical/bin column."""
    out = df.copy()
    out['Fare Interval'] = pd.cut(
        out['Fare'], bins=cuts, labels=labels, include_lowest=True, right=True
    )
    out['Fare Interval'] = out['Fare Interval'].astype(float)
    return out


def add_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined 'Sex_Pclass' feature."""
    out = df.copy()
    out['Sex_Pclass'] = out.apply(
        lambda r: (str(r.get('Sex', ''))[:1] or '?').upper() + "_C" + str(int(r['Pclass'])),
        axis=1,
    )
    return out


def parse_names_row(row) -> pd.Series:
    """Parse a Titanic passenger 'Name' field into components."""
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").strip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0].strip()
            maiden_name = split_text[1].rstrip(")").strip()
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text.strip()
            return pd.Series([family_name, title, given_name, None])
    except Exception:
        return pd.Series([None, None, None, None])


def add_parsed_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed name components to dataframe."""
    out = df.copy()
    out[["Family Name", "Title", "Given Name", "Maiden Name"]] = out.apply(
        parse_names_row, axis=1
    )
    return out


def copy_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate the Title column as Titles."""
    out = df.copy()
    out["Titles"] = out["Title"]
    return out


def add_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """Map Family Size into Family Type categorical values."""
    out = df.copy()
    out['Family Type'] = out['Family Size']
    out.loc[out['Family Size'] == 1, 'Family Type'] = 'Single'
    out.loc[(out['Family Size'] > 1) & (out['Family Size'] < 5), 'Family Type'] = 'Small'
    out.loc[out['Family Size'] >= 5, 'Family Type'] = 'Large'
    return out


def unify_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize rare titles."""
    out = df.copy()
    if 'Titles' not in out.columns:
        return out
    out['Titles'] = out['Titles'].replace({'Mlle.': 'Miss.', 'Ms.': 'Miss.', 'Mme.': 'Mrs.'})
    rare = [
        'Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.',
        'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'
    ]
    out['Titles'] = out['Titles'].replace(rare, 'Rare')
    return out


def encode_sex_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Sex as numeric 0/1."""
    out = df.copy()
    out['Sex'] = out['Sex'].map({'female': 1, 'male': 0}).astype('Int64')
    return out
