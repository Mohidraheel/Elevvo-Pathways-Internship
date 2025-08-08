import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


data=pd.read_csv("D:\\Fast\\python\\Intern\\task4\\loan_approval_dataset.csv")
print(data.head())
print(data.shape)

print(data.isnull().sum())
print(data.columns)
data.columns = data.columns.str.strip()
data['loan_status'] = data['loan_status'].map({' Approved': 1, ' Rejected': 0})
data['education']= data['education'].map({' Graduate': 1, ' Not Graduate': 0})
data['self_employed'] = data['self_employed'].map({' Yes': 1, ' No': 0})
print(data.head())

plt.figure(figsize=(8, 5))
sns.countplot(x='loan_status', data=data, palette='Set2')
plt.title('Loan Approval Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='education', hue='loan_status', data=data, palette='Set2')
plt.title('Loan Approval Status by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Loan Status', loc='upper right', labels=['Rejected', 'Approved'])
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='self_employed', hue='loan_status', data=data, palette='Set2')
plt.title('Loan Approval Status by Self-Employment Status')
plt.xlabel('Self-Employment Status')
plt.ylabel('Count')
plt.legend(title='Loan Status', loc='upper right', labels=['Rejected', 'Approved'])
plt.show()

X= data.drop(columns=['loan_status'], axis=1)
y= data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote=SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())

model=XGBClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
