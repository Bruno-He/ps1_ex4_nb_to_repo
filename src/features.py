import pandas as pd
import numpy as np


def create_family_size(df):
    """
    Create family size feature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SibSp and Parch columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Family Size column
    """
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_age_intervals(df):
    """
    Create age interval feature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Age column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Age Interval column
    """
    df = df.copy()
    df["Age Interval"] = 0.0
    df.loc[df['Age'] <= 16, 'Age Interval'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[df['Age'] > 64, 'Age Interval'] = 4
    return df


def create_fare_intervals(df):
    """
    Create fare interval feature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Fare column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Fare Interval column
    """
    df = df.copy()
    df['Fare Interval'] = 0.0
    df.loc[df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval'] = 2
    df.loc[df['Fare'] > 31, 'Fare Interval'] = 3
    return df


def create_sex_pclass_feature(df):
    """
    Create sex and passenger class combination feature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Sex and Pclass columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Sex_Pclass column
    """
    df = df.copy()
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df


def create_family_type(df):
    """
    Create family type feature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Family Size column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Family Type column
    """
    df = df.copy()
    df["Family Type"] = df["Family Size"]
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[(df["Family Size"] >= 5), "Family Type"] = "Large"
    return df


def standardize_titles(df):
    """
    Standardize title features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Title column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Titles column, titles standardized
    """
    df = df.copy()
    df["Titles"] = df["Title"]
    
    # 统一Miss相关头衔
    df['Titles'] = df['Titles'].replace('Mlle.', 'Miss.')
    df['Titles'] = df['Titles'].replace('Ms.', 'Miss.')
    
    # 统一Mrs相关头衔
    df['Titles'] = df['Titles'].replace('Mme.', 'Mrs.')
    
    # 统一稀有头衔
    rare_titles = ['Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 
                   'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.']
    df['Titles'] = df['Titles'].replace(rare_titles, 'Rare')
    
    return df


def parse_names(row):
    """
    Parse name string to extract family name, title, given name and maiden name
    
    Parameters:
    -----------
    row : pandas.Series
        Row data containing Name column
        
    Returns:
    --------
    pandas.Series
        Series containing [Family Name, Title, Given Name, Maiden Name]
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception parsing name '{row['Name']}': {ex}")
        return pd.Series([None, None, None, None])


def extract_name_features(df):
    """
    Extract features from names
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Name column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Family Name, Title, Given Name, Maiden Name columns
    """
    df = df.copy()
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df


def combine_datasets(train_df, test_df):
    """
    Combine training and test datasets with dataset identifier
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame
    test_df : pandas.DataFrame
        Test DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with set column identifying data source
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df
