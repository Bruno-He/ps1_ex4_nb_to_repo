import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

from src.data import load_data
from src.features import (create_family_size, create_age_intervals, create_fare_intervals, 
                     create_family_type)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
DIRECTORY = "figures"

def basic_setup():

    train_df, test_df= load_data(TRAIN_PATH, TEST_PATH)
    # Set plot style
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Prepare data: Create family size feature (using only training data for analysis since test set doesn't have Survived labels)
    train_data = train_df
    train_data = create_family_size(train_data)
    train_data = create_family_type(train_data)
    return train_data
    


def draw_gender_and_survival_rate(train_data):
    # Gender and Survival Rate Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Survival count by gender
    sns.countplot(data=train_data, x='Sex', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival Count by Gender', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gender', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].legend(['Did not survive', 'Survived'], title='Survival Status')
    axes[0].grid(axis='y', alpha=0.3)

    # Right plot: Survival rate by gender
    survival_by_sex = train_data.groupby('Sex')['Survived'].agg(['mean', 'count'])
    survival_by_sex.columns = ['Survival Rate', 'Total Count']
    survival_by_sex['Survival Rate'] = survival_by_sex['Survival Rate'] * 100

    bars = axes[1].bar(survival_by_sex.index, survival_by_sex['Survival Rate'], 
                        color=['#3498db', '#e91e63'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Gender', fontsize=12)
    axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (idx, row) in enumerate(survival_by_sex.iterrows()):
        axes[1].text(i, row['Survival Rate'] + 2, f"{row['Survival Rate']:.1f}%", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[1].text(i, row['Survival Rate'] - 5, f"n={int(row['Total Count'])}", 
                    ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    my_file_name = "Gender and Survival Rate"
    save_path = os.path.join(DIRECTORY, f"{my_file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def draw_age_and_survival_rate(train_data):
    # Age and Survival Rate Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Create age intervals
    train_data_age = train_data.copy()
    train_data_age = create_age_intervals(train_data_age)
    age_labels = ['0-16 years', '17-32 years', '33-48 years', '49-64 years', '65+ years']
    train_data_age['Age Interval Label'] = train_data_age['Age Interval'].map({
        0: '0-16 years', 1: '17-32 years', 2: '33-48 years', 3: '49-64 years', 4: '65+ years'
    })

    # Left: Age distribution histogram (grouped by survival status)
    train_data_with_age = train_data_age.dropna(subset=['Age'])
    sns.histplot(data=train_data_with_age, x='Age', hue='Survived', bins=30, 
                kde=True, ax=axes[0], palette=['#e74c3c', '#2ecc71'], alpha=0.6)
    axes[0].set_title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].legend(['Did not survive', 'Survived'], title='Survival Status')
    axes[0].grid(axis='y', alpha=0.3)

    # Right: Survival rate by age interval
    survival_by_age = train_data_age.groupby('Age Interval Label')['Survived'].agg(['mean', 'count'])
    survival_by_age.columns = ['Survival Rate', 'Total Count']
    survival_by_age = survival_by_age.reindex(age_labels)
    survival_by_age['Survival Rate'] = survival_by_age['Survival Rate'] * 100

    bars = axes[1].bar(range(len(survival_by_age)), survival_by_age['Survival Rate'], 
                        color=plt.cm.viridis(np.linspace(1, len(survival_by_age))), 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Survival Rate by Age Interval', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Age Interval', fontsize=12)
    axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[1].set_xticks(range(len(survival_by_age)))
    axes[1].set_xticklabels(survival_by_age.index, rotation=45, ha='right')
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)

# Add value labels
    for i, (idx, row) in enumerate(survival_by_age.iterrows()):
        if not pd.isna(row['Survival Rate']):
            axes[1].text(i, row['Survival Rate'] + 2, f"{row['Survival Rate']:.1f}%", 
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    my_file_name = "Age and Survival Rate"
    save_path = os.path.join(DIRECTORY, f"{my_file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def draw_passenger_class_and_survival_rate(train_data):
    # Passenger Class and Survival Rate Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left plot: Survival count by passenger class
    pclass_order = [1, 2, 3]
    sns.countplot(data=train_data, x='Pclass', hue='Survived', ax=axes[0], 
                order=pclass_order, palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival Count by Passenger Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Passenger Class (1=First, 2=Second, 3=Third)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].legend(['Did not survive', 'Survived'], title='Survival Status')
    axes[0].grid(axis='y', alpha=0.3)

    # Middle plot: Survival rate by passenger class
    survival_by_pclass = train_data.groupby('Pclass')['Survived'].agg(['mean', 'count'])
    survival_by_pclass.columns = ['Survival Rate', 'Total Count']
    survival_by_pclass['Survival Rate'] = survival_by_pclass['Survival Rate'] * 100

    colors = ['#f39c12', '#3498db', '#e74c3c']  # Gold, blue, red representing three classes
    bars = axes[1].bar(survival_by_pclass.index, survival_by_pclass['Survival Rate'], 
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Passenger Class', fontsize=12)
    axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[1].set_xticks(survival_by_pclass.index)
    axes[1].set_xticklabels(['First Class', 'Second Class', 'Third Class'])
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for idx, row in survival_by_pclass.iterrows():
        axes[1].text(idx, row['Survival Rate'] + 2, f"{row['Survival Rate']:.1f}%", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[1].text(idx, row['Survival Rate'] - 5, f"n={int(row['Total Count'])}", 
                    ha='center', va='top', fontsize=9, style='italic')


    # Right plot: Interaction effect between passenger class and gender
    survival_by_pclass_sex = train_data.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100
    survival_by_pclass_sex = survival_by_pclass_sex.reset_index()
    survival_by_pclass_sex.columns = ['Pclass', 'Sex', 'Survival Rate']

    x = np.arange(len(pclass_order))
    width = 0.35
    for i, sex in enumerate(['female', 'male']):
        values = [survival_by_pclass_sex[(survival_by_pclass_sex['Pclass']==p) & 
                                        (survival_by_pclass_sex['Sex']==sex)]['Survival Rate'].values[0] 
                for p in pclass_order]
        axes[2].bar(x + i*width, values, width, label='Female' if sex=='female' else 'Male', 
                    alpha=0.7, edgecolor='black', linewidth=1)
    axes[2].set_title('Interaction: Passenger Class and Gender', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Passenger Class', fontsize=12)
    axes[2].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[2].set_xticks(x + width / 2)
    axes[2].set_xticklabels(['First Class', 'Second Class', 'Third Class'])
    axes[2].legend()
    axes[2].set_ylim([0, 100])
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    my_file_name = "Passenger Class and Survival Rate"
    save_path = os.path.join(DIRECTORY, f"{my_file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def draw_family_size_and_survival_rate(train_data):
        # Family Size and Survival Rate Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Right: Survival rate by family size
    survival_by_family = train_data.groupby('Family Size')['Survived'].agg(['mean', 'count'])
    survival_by_family.columns = ['Survival Rate', 'Total Count']
    survival_by_family['Survival Rate'] = survival_by_family['Survival Rate'] * 100

    bars = axes[1].bar(survival_by_family.index, survival_by_family['Survival Rate'], 
                        color=plt.cm.coolwarm(np.linspace(0, 1, len(survival_by_family))), 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Family Size (number of people)', fontsize=12)
    axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for idx, row in survival_by_family.iterrows():
        axes[1].text(idx, row['Survival Rate'] + 2, f"{row['Survival Rate']:.1f}%", 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1].text(idx, row['Survival Rate'] - 5, f"n={int(row['Total Count'])}", 
                        ha='center', va='top', fontsize=8, style='italic')

    # Left: Family type and survival rate
    survival_by_family_type = train_data.groupby('Family Type')['Survived'].agg(['mean', 'count'])
    survival_by_family_type.columns = ['Survival Rate', 'Total Count']
    survival_by_family_type['Survival Rate'] = survival_by_family_type['Survival Rate'] * 100
    # Order: Single, Small, Large
    type_order = ['Single', 'Small', 'Large']
    survival_by_family_type = survival_by_family_type.reindex(type_order)

    type_labels = ['Single', 'Small Family\n(2-4 people)', 'Large Family\n(5+ people)']
    bars = axes[0].bar(range(len(survival_by_family_type)), survival_by_family_type['Survival Rate'], 
                        color=['#95a5a6', '#3498db', '#e67e22'], 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Survival Rate by Family Type', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Family Type', fontsize=12)
    axes[0].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[0].set_xticks(range(len(survival_by_family_type)))
    axes[0].set_xticklabels(type_labels)
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(survival_by_family_type.iterrows()):
        axes[0].text(i, row['Survival Rate'] + 2, f"{row['Survival Rate']:.1f}%", 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    my_file_name = "Family Size and Survival Rate"
    save_path = os.path.join(DIRECTORY, f"{my_file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def draw_fare_and_survival_rate(train_data):
    # Fare and Survival Rate Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Handle missing values
    train_data_fare = train_data.copy()
    train_data_fare = create_fare_intervals(train_data_fare)
    fare_labels = ['Low Fare\n(≤7.91)', 'Medium-Low Fare\n(7.91-14.45)', 'Medium-High Fare\n(14.45-31)', 'High Fare\n(>31)']
    train_data_fare['Fare Interval Label'] = train_data_fare['Fare Interval'].map({
        0: 'Low Fare\n(≤7.91)', 1: 'Medium-Low Fare\n(7.91-14.45)', 2: 'Medium-High Fare\n(14.45-31)', 3: 'High Fare\n(>31)'
    })

    # Right: Survival rate by fare interval
    survival_by_fare = train_data_fare.groupby('Fare Interval Label')['Survived'].agg(['mean', 'count'])
    survival_by_fare.columns = ['Survival Rate', 'Total Count']
    survival_by_fare['Survival Rate'] = survival_by_fare['Survival Rate'] * 100
    # Order
    fare_order = ['Low Fare\n(≤7.91)', 'Medium-Low Fare\n(7.91-14.45)', 'Medium-High Fare\n(14.45-31)', 'High Fare\n(>31)']
    survival_by_fare = survival_by_fare.reindex(fare_order)

    bars = axes[1].bar(range(len(survival_by_fare)), survival_by_fare['Survival Rate'], 
                        color=plt.cm.plasma(np.linspace(0, 1, len(survival_by_fare))), 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Survival Rate by Fare Interval', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Fare Interval', fontsize=12)
    axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[1].set_xticks(range(len(survival_by_fare)))
    axes[1].set_xticklabels(survival_by_fare.index, rotation=0, ha='center')
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(survival_by_fare.iterrows()):
        if not pd.isna(row['Survival Rate']):
            axes[1].text(i, row['Survival Rate'] + 2, f"{row['Survival Rate']:.1f}%", 
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            axes[1].text(i, row['Survival Rate'] - 5, f"n={int(row['Total Count'])}", 
                            ha='center', va='top', fontsize=8, style='italic')

    # Left: Boxplot - Fare distribution for survived vs not survived passengers
    sns.boxplot(data=train_data_fare, x='Survived', y='Fare', ax=axes[0], 
                palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Fare Distribution by Survival Status (Boxplot)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Survival Status', fontsize=12)
    axes[0].set_ylabel('Fare', fontsize=12)
    axes[0].set_xticklabels(['Did not survive', 'Survived'])
    axes[0].set_yscale('log')  # Use log scale because fare distribution is skewed
    axes[0].grid(axis='y', alpha=0.3)


    plt.tight_layout()
    my_file_name = "Fare and Survival Rate"
    save_path = os.path.join(DIRECTORY, f"{my_file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def draw_all_pictures(train_data):
    draw_gender_and_survival_rate(train_data)
    draw_age_and_survival_rate(train_data)
    draw_passenger_class_and_survival_rate(train_data)
    draw_family_size_and_survival_rate(train_data)
    draw_fare_and_survival_rate(train_data)