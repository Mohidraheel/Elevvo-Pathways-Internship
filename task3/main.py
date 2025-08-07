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

data=pd.read_csv("D:\\Fast\\python\\Intern\\task3\\covtype1.csv")
print(data.head())
data['Cover_Type']=data['Cover_Type']-1
print(data['Cover_Type'].value_counts())
cmap=sns.color_palette("Set2", as_cmap=True)(np.arange(7))
plt.figure(figsize=(10, 8))
plt.pie(
    data['Cover_Type'].value_counts().values,
    labels=data['Cover_Type'].value_counts().keys(),
    autopct='%1.1f%%',
    colors=cmap
)
plt.title('Distribution of Cover Types')
plt.show()

X=data.drop(columns=['Cover_Type'], axis=1)
y=data['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar= StandardScaler()
X_train =pd.DataFrame(scalar.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns)
model = XGBClassifier(objective='multi:softmax', num_class=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[-15:][::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45)
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()
