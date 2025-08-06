import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("D:\\Fast\\python\\project\\CustomerSegmentation\\Mall_Customers.csv")
print(data.head())
print(data.isnull().sum())
print(data.describe())


X = data.iloc[:, [3, 4]].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X_scaled)
print(Y)


plt.figure(figsize=(8, 8))
plt.scatter(X_scaled[Y == 0, 0], X_scaled[Y == 0, 1], s=50, c='Green', label='Cluster 1')
plt.scatter(X_scaled[Y == 1, 0], X_scaled[Y == 1, 1], s=50, c='Red', label='Cluster 2')
plt.scatter(X_scaled[Y == 2, 0], X_scaled[Y == 2, 1], s=50, c='Blue', label='Cluster 3')
plt.scatter(X_scaled[Y == 3, 0], X_scaled[Y == 3, 1], s=50, c='Purple', label='Cluster 4')
plt.scatter(X_scaled[Y == 4, 0], X_scaled[Y == 4, 1], s=50, c='Orange', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='Yellow', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()


data['Cluster'] = Y
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())


sns.countplot(x='Cluster', data=data, palette='viridis')
plt.title('Number of Customers per Cluster')
plt.show()
