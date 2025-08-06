import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv("D:\\Fast\\python\\Intern\\task1\\StudentPerformanceFactors.csv")
print(data)
print(data.head())
print(data.isnull().sum())
data = data.dropna()
print(data.duplicated().sum())
print(data.info())
print(data.select_dtypes(include='object').columns)


#replacing the categorical values with numerical values
data['Parental_Involvement'] = data['Parental_Involvement'].map({'High': 2, 'Medium': 1, 'Low': 0})
data['Access_to_Resources'] = data['Access_to_Resources'].map({'High': 2, 'Medium': 1, 'Low': 0})
data['Peer_Influence'] = data['Peer_Influence'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0})
data['Parental_Education_Level'] = data['Parental_Education_Level'].map({'High School': 0, 'College': 1, 'Postgraduate': 2})
data['Distance_from_Home'] = data['Distance_from_Home'].map({'Near': 0, 'Moderate': 1, 'Far': 2})
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})
data['Learning_Disabilities'] = data['Learning_Disabilities'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['School_Type'] = data['School_Type'].map({'Public': 0, 'Private': 1})
data['Motivation_Level'] = data['Motivation_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Internet_Access'] = data['Internet_Access'].map({'No': 0, 'Yes': 1})
data['Family_Income'] = data['Family_Income'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Teacher_Quality'] = data['Teacher_Quality'].map({'Low': 0, 'Medium': 1, 'High': 2})


categorical_columns = [
    'Parental_Involvement',
    'Access_to_Resources',
    'Peer_Influence',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Extracurricular_Activities',
    'Learning_Disabilities',
    'Gender',
    'School_Type',
    'Motivation_Level',
    'Internet_Access',
    'Family_Income'
    'Teacher_Quality'
]

for col in categorical_columns:
    counts = data[col].value_counts().sort_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f'Value Counts for {col}')
    plt.ylabel('Count')
    plt.show()




correlation_matrix = data.select_dtypes(include=['number']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


X=data.drop(['Exam_Score'], axis=1)
y=data['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred.round()))

print("Bonus")
polynomialfeatures = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = polynomialfeatures.fit_transform(X_train)
X_test_poly = polynomialfeatures.transform(X_test)

# Train polynomial model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions
y_pred_poly = poly_model.predict(X_test_poly)
y_train_pred_poly = poly_model.predict(X_train_poly)

# Model coefficients
print(f"Polynomial Model Coefficients:")
print(f"Intercept: {poly_model.intercept_:.4f}")
print(f"Coefficients: {poly_model.coef_}")


meanpoly = mean_squared_error(y_test, y_pred_poly)
rmeanpoly = np.sqrt(meanpoly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"\nPolynomial Regression Performance:")
print(f"Mean Squared Error: {meanpoly:.4f}")
print(f"Root Mean Squared Error: {rmeanpoly:.4f}")
print(f"Mean Absolute Error: {mae_poly:.4f}")
print(f"R² Score: {r2_poly:.4f}")
 
