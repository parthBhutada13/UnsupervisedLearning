# #Importing Libraries


#mall customers clustering analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# #Data Preprocessing


#load dataset
df = pd.read_csv('/content/mallCustomers.csv')
print("dataset shape:", df.shape)
print("first 5 rows:")
print(df.head())

#data preprocessing
df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)

#check for missing values
print("missing values:")
print(df.isna().sum())

#encode gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

#eda (plotting graphs against CustomerID for better understanding)

import matplotlib.pyplot as plt
features = ['Age', 'Income', 'Score', 'Gender']
titles = ['Age', 'Income', 'Score', 'Gender (0=Female, 1=Male)']

plt.figure(figsize=(12, 10))

for idx, feature in enumerate(features):
    plt.subplot(2, 2, idx + 1)
    plt.scatter(df['CustomerID'], df[feature], color='blue', edgecolor='gray', alpha=0.4)
    plt.title(f'Customer ID vs {titles[idx]}', fontsize=12)
    plt.xlabel('Customer ID')
    plt.ylabel(titles[idx])
    plt.grid(True)

plt.tight_layout()
plt.show()

#drop customer id (not relevant for clustering)
df = df.drop('CustomerID', axis=1)

#scale features using minmax scaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#exploratory data analysis
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("correlation matrix of scaled features")
plt.tight_layout()
plt.show()
