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

# #Functions Definition


def evaluate_clustering(data, labels, method_name):
    if len(set(labels)) <= 1:
        return None, None

    sil_score = silhouette_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)

    print(f"{method_name}:")
    print(f" silhouette score: {sil_score:.3f}")
    print(f" calinski-harabasz score: {ch_score:.3f}")

    return sil_score, ch_score

def plot_clusters(data, labels, title, dim_names=['dim1', 'dim2']):
    plt.figure(figsize=(8, 6))
    df_plot = pd.DataFrame(data, columns=dim_names)
    df_plot['cluster'] = labels

    sns.scatterplot(data=df_plot, x=dim_names[0], y=dim_names[1], hue='cluster', palette='Set1', s=60, alpha=0.4)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#find optimal k using elbow method
def find_optimal_k(data, max_k=10):
    inertias = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=64, n_init='auto')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.title('elbow method for optimal k')
    plt.xlabel('number of clusters (k)')
    plt.ylabel('inertia')
    plt.grid(True, alpha=0.3)
    plt.show()

    return k_range, inertias

# #Model Building


# ##PCA + Kmeans


#pca + k-means
print("pca + k-means clustering")
pca = PCA(n_components = 2, random_state = 64)
pca_data = pca.fit_transform(df_scaled)

#find optimal k
find_optimal_k(pca_data)

results = {}

#apply k-means with k=3 (from elbow method)
kmeans = KMeans(n_clusters = 3, random_state = 64, n_init='auto')
kmeans_labels = kmeans.fit_predict(pca_data)

plot_clusters(pca_data, kmeans_labels, 'pca + k-means clustering', ['pc1', 'pc2'])
sil_pca_km, ch_pca_km = evaluate_clustering(pca_data, kmeans_labels, 'pca + k-means')
results['pca + k-means'] = {'silhouette': sil_pca_km, 'calinski_harabasz': ch_pca_km}
