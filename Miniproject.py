# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:05:25 2025

@author: shaba
"""
## Importing the different libraries to use

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
importances = []

## Loading the dataset

ts = pd.read_csv("C:/Users/shaba/Desktop/shad/customer_churn_prediction_dataset.csv")
print(ts)

##showing the first five and the last five
 
ts.head()
ts.tail()

## Data Cleaning
## Drop customerID

ts.drop(columns=["customerID"], inplace=True)
print(ts)

#Exploratory Data Analysis

##General information about the dataset
ts.info()

## Summary statistics for numerical features
ts.describe()

## Check for missing values
ts.isnull().sum()

## creating a labelEncoder
Label_pre = LabelEncoder()
data_cols = ts.select_dtypes(exclude=['int','float']).columns
label_col =list(data_cols)

## Applying encoder
ts[label_col] = ts[label_col].apply(lambda col:Label_pre.fit_transform(col))

## Saving dataset with labelEncoder
ts.to_csv("dataset_LabelEncoder.csv")
Label_pre
ts.head()
ts.tail()

## Visualize distribution of churn
sns.countplot(x='Churn', data=ts)  # Assuming 'Churn' column contains binary values (0 - No churn, 1 - Churn)

## Visualize correlation between numerical features
sns.heatmap(ts.corr(), annot=True, cmap='coolwarm')

## filling
ts['TotalCharges'].fillna(ts['TotalCharges'].median(), inplace=True)


## Visualize churn with key demographic features (e.g., gender, tenure)
sns.countplot(x='gender', hue='Churn', data=ts)  


## Convert 'TotalCharges' to numeric (some may be blank strings)
ts['TotalCharges'] = pd.to_numeric(ts['TotalCharges'], errors='coerce')

## Fill missing values in 'TotalCharges' with the median
ts['TotalCharges'] = ts['TotalCharges'].fillna(ts['TotalCharges'].median())

## List of categorical columns
categorical_cols = ts.select_dtypes(include=['object']).columns.tolist()


## Visualize correlation (numerical features only)
plt.figure(figsize=(10, 6))
sns.heatmap(ts.corr(), annot=True, cmap='coolwarm')
plt.subplot(1, 3, 3)
sns.countplot(x="SeniorCitizen", hue="Churn", data=ts)

              
## Data Processing

## Handle missing values (if any) using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
ts = pd.DataFrame(imputer.fit_transform(ts), columns=ts.columns)


## Encode categorical features using LabelEncoder or OneHotEncoding
label_encoder = LabelEncoder()

## Assuming categorical columns like 'gender', 'Contract', 'PaymentMethod', etc.
ts['gender'] = label_encoder.fit_transform(ts['gender'])
ts['PaymentMethod'] = label_encoder.fit_transform(ts['PaymentMethod'])
ts['Contract'] = label_encoder.fit_transform(ts['Contract'])

## Feature scaling (for models like Logistic Regression, Random Forest, etc.)
scaler = StandardScaler()
## Convert TotalCharges to numeric and handle missing values
ts['TotalCharges'] = pd.to_numeric(ts['TotalCharges'], errors='coerce')
ts['TotalCharges'].fillna(ts['TotalCharges'].median(), inplace=True)
categorical_cols = ts.select_dtypes(include=['object']).columns.tolist()
numerical_cols = ts.select_dtypes(include=['int64', 'float64']).columns.tolist()

## Remove target variable from numerical columns
numerical_cols.remove('Churn') if 'Churn' in numerical_cols else None
scaled_features = scaler.fit_transform(ts.drop('Churn', axis=1))  # Scaling all features except the target variable


## Separate the target variable (Churn) from features
X = ts.drop('Churn', axis=1)
y = ts['Churn']


## Train-Test Splint

## Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Building and training machine learning models

## Initialize Logistic Regression model
log_reg = LogisticRegression()

## Train the model
log_reg.fit(X_train, y_train)

## Predict on test data
y_pred_log_reg = log_reg.predict(X_test)

## Model evaluation
print('Logistic Regression Model Evaluation:')
print('Accuracy:', accuracy_score(y_test, y_pred_log_reg))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_log_reg))
print('Classification Report:')
print(classification_report(y_test, y_pred_log_reg))

## Initialize Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

## Train the model
rf.fit(X_train, y_train)

## Predict on test data
y_pred_rf = rf.predict(X_test)

## Model evaluation
print('Random Forest Model Evaluation:')
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))
print('Classification Report:')
print(classification_report(y_test, y_pred_rf))

## Feature importance
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

## Plot feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importance - Random Forest')
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.show()

## Gradient Boosting
## Initialize Gradient Boosting model
gb = GradientBoostingClassifier(random_state=42)

## Train the model
gb.fit(X_train, y_train)

## Predict on test data
y_pred_gb = gb.predict(X_test)

## Model evaluation
print('Gradient Boosting Model Evaluation:')
print('Accuracy:', accuracy_score(y_test, y_pred_gb))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_gb))
print('Classification Report:')
print(classification_report(y_test, y_pred_gb))

## AUC-ROC score for each model
roc_auc_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
roc_auc_gb = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

print(f'AUC-ROC (Logistic Regression): {roc_auc_log_reg}')
print(f'AUC-ROC (Random Forest): {roc_auc_rf}')
print(f'AUC-ROC (Gradient Boosting): {roc_auc_gb}')

## THE END OF MY TRIALS