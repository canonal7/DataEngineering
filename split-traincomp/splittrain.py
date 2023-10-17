import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import classification_report

np.random.seed(42)

predictors_cov = ['Gender', 'Age', 'Scholarship', 'Hipertension','Diabetes', 'Alcoholism','lead_days', 'handicap_boolean']
target = 'No-show'

X_train, X_test, y_train, y_test = train_test_split(df[predictors_cov], df[target], test_size=0.2, random_state=42)

sm = SMOTENC(random_state=42, categorical_features = [True])
X_train_smote_cov, y_train_smote_cov = sm.fit_resample(X_train, y_train)

print('Original dataset shape')
print('False:', sum(y_train == False))
print('True: ', sum(y_train == True))
print('Resampled dataset COV shape with SMOTE')
print('False:', sum(y_train_smote_cov == False))
print('True: ', sum(y_train_smote_cov == True))

scaler = MinMaxScaler()
X_train_smote_cov[['Age', 'lead_days']] = scaler.fit_transform(X_train_smote_cov[['Age','lead_days']])
X_train_ros_cov[['Age', 'lead_days']] = scaler.fit_transform(X_train_ros_cov[['Age','lead_days']])
X_train_rus_cov[['Age', 'lead_days']] = scaler.fit_transform(X_train_rus_cov[['Age','lead_days']])