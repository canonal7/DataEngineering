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

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['Gender'] = df['Gender'].map({'F': True, 'M': False})
df['No-show'] = df['No-show'].map({'Yes': True, 'No': False})
df['Scholarship'] = df['Scholarship'].astype(bool)
df['Diabetes'] = df['Diabetes'].astype(bool)
df['Hipertension'] = df['Hipertension'].astype(bool)
df['Alcoholism'] = df['Alcoholism'].astype(bool)
df['SMS_received'] = df['SMS_received'].astype(bool)
df['handicap_boolean'] = df['Handcap'].replace([2, 3, 4], 1).astype(bool)

df['only_date_appointment_day'] = df['AppointmentDay'].dt.date
df['only_date_scheduled_day'] = df['ScheduledDay'].dt.date
df['lead_days'] = (df['only_date_appointment_day'] - df['only_date_scheduled_day']).dt.days.astype(np.int64)
df = df.drop('only_date_appointment_day', axis = 1)
df = df.drop('only_date_scheduled_day', axis = 1)

df = df[df['Age'] >= 0]
df = df[df['Age'] <= 100]
df = df[df['lead_days'] >= 0]
df = df[df['lead_days'] != 398]
df = df.drop('SMS_received', axis = 1)
df = df.drop('Neighbourhood', axis = 1)