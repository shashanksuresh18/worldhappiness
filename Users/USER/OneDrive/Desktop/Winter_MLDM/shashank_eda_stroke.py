import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#This complete process is done by Shashank Suresh
# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')  

print("First few rows of the dataset:")
print(df.head())

df['hypertension'] = df['hypertension'].astype('object')
df['heart_disease'] = df['heart_disease'].astype('object')
df['stroke'] = df['stroke'].astype('object')

df.drop('id', axis=1, inplace=True)

print("Summary statistics:")
print(df.describe())

print("Data types:")
print(df.dtypes)

print("Checking for NaN or null values:")
print(df.isnull().sum())

# Fill missing values
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for column in categorical_cols:
    df[column].fillna(df[column].mode()[0], inplace=True)

print("Checking for NaN or null values after filling:")
print(df.isnull().sum())

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease', 'stroke']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Detect and replace outliers
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
    return outliers

def replace_outliers(df, strategy='mean'):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        outliers = detect_outliers(df, col)
        if strategy == 'mean':
            replacement = df[col].mean()
        elif strategy == 'median':
            replacement = df[col].median()
        elif strategy == 'mode':
            replacement = df[col].mode()[0]
        df.loc[outliers, col] = replacement
    return df

df_treated = df.copy()
df_treated = replace_outliers(df_treated, strategy='mean')

print(f"Dataset shape after replacing outliers: {df_treated.shape}")

print("Class distribution in target 'stroke':")
print(df_treated['stroke'].value_counts())
