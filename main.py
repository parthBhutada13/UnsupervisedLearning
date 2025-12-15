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

# ##TSNE + Kmeans


#tsne + k-means
print("t-sne + k-means clustering")

best_tsne_score = -1
best_tsne_params = {}

perplexities = [5, 10, 20, 30, 40, 50]
n_iters = [500, 1000, 2000, 3000]

for perp in perplexities:
    for n_iter in n_iters:
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter, random_state=64)
        tsne_data = tsne.fit_transform(df_scaled)

        kmeans = KMeans(n_clusters=3, random_state=64, n_init='auto')
        labels = kmeans.fit_predict(tsne_data)

        score = silhouette_score(tsne_data, labels)
        if score > best_tsne_score:
            best_tsne_score = score
            best_tsne_params = {'perplexity': perp, 'n_iter': n_iter,
                               'data': tsne_data, 'labels': labels}

plot_clusters(best_tsne_params['data'], best_tsne_params['labels'], 't-sne + k-means clustering')
sil_tsne_km, ch_tsne_km = evaluate_clustering(best_tsne_params['data'], best_tsne_params['labels'], 't-sne + k-means')

results['t-sne + k-means'] = {'silhouette': sil_tsne_km, 'calinski_harabasz': ch_tsne_km}

# ##PCA + DBSCAN


#pca + dbscan
print("pca + dbscan clustering")
best_dbscan_score = -1
best_dbscan_params = {}

for eps in [0.1, 0.2, 0.3]:
    for min_samples in [3, 5, 7]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pca_data)

        if len(set(labels)) <= 1 or (len(set(labels)) == 2 and -1 in labels):
            continue

        score = silhouette_score(pca_data, labels)
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_dbscan_params = {'eps': eps, 'min_samples': min_samples, 'labels': labels}

if best_dbscan_params:
    plot_clusters(pca_data, best_dbscan_params['labels'], 'pca + dbscan clustering', ['pc1', 'pc2'])
    sil_pca_db, ch_pca_db = evaluate_clustering(pca_data, best_dbscan_params['labels'], 'pca + dbscan')
    results['pca + dbscan'] = {'silhouette': sil_pca_db, 'calinski_harabasz': ch_pca_db}

# ##TSNE + DBSCAN


#tsne + dbscan
print("t-sne + dbscan clustering")
tsne_data = best_tsne_params['data']

best_tsne_dbscan_score = -1
best_tsne_dbscan_params = {}

for eps in [2, 4, 6]:
    for min_samples in [8, 12, 16]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(tsne_data)

        if len(set(labels)) <= 1:
            continue

        score = silhouette_score(tsne_data, labels)
        if score > best_tsne_dbscan_score:
            best_tsne_dbscan_score = score
            best_tsne_dbscan_params = {'labels': labels}

if best_tsne_dbscan_params:
    plot_clusters(tsne_data, best_tsne_dbscan_params['labels'], 't-sne + dbscan clustering')
    sil_tsne_db, ch_tsne_db = evaluate_clustering(tsne_data, best_tsne_dbscan_params['labels'],
                                                 't-sne + dbscan')
    results['t-sne + dbscan'] = {'silhouette': sil_tsne_db, 'calinski_harabasz': ch_tsne_db}

# ##PCA + Hierarchical Clustering


#pca + hierarchical clustering
print("pca + hierarchical clustering")
best_hier_score = -1
best_hier_params = {}

for n_clusters in [2, 3, 4]:
    for linkage_method in ['ward', 'complete', 'average']:
        hier = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hier.fit_predict(pca_data)

        score = silhouette_score(pca_data, labels)
        if score > best_hier_score:
            best_hier_score = score
            best_hier_params = {'n_clusters': n_clusters, 'linkage': linkage_method, 'labels': labels}

plot_clusters(pca_data, best_hier_params['labels'], 'pca + hierarchical clustering', ['pc1', 'pc2'])
sil_pca_hier, ch_pca_hier = evaluate_clustering(pca_data, best_hier_params['labels'], 'pca + hierarchical')
results['pca + hierarchical'] = {'silhouette': sil_pca_hier, 'calinski_harabasz': ch_pca_hier}

#plot dendrogram
linkage_matrix = linkage(pca_data, method=best_hier_params['linkage'])
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('dendrogram - pca + hierarchical clustering')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ##TSNE + Hierarchical Clustering


#tsne + hierarchical clustering
print("t-sne + hierarchical clustering")
best_tsne_hier_score = -1
best_tsne_hier_params = {}

for n_clusters in [3, 4, 5]:
    for linkage_method in ['ward', 'complete', 'average']:
        hier = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hier.fit_predict(tsne_data)

        score = silhouette_score(tsne_data, labels)
        if score > best_tsne_hier_score:
            best_tsne_hier_score = score
            best_tsne_hier_params = {'labels': labels, 'linkage': linkage_method}

plot_clusters(tsne_data, best_tsne_hier_params['labels'], 't-sne + hierarchical clustering')
sil_tsne_hier, ch_tsne_hier = evaluate_clustering(tsne_data, best_tsne_hier_params['labels'], 't-sne + hierarchical')
results['t-sne + hierarchical'] = {'silhouette': sil_tsne_hier, 'calinski_harabasz': ch_tsne_hier}

#plot dendrogram
linkage_matrix = linkage(tsne_data, method=best_tsne_hier_params['linkage'])
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('dendrogram - t-sne + hierarchical clustering')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# #Final Comparison


#final comparison
print("" + "=" * 24)
print("final results comparison")
print("=" * 24)

results_df = pd.DataFrame(results).T
print(results_df)

#find best performing method
best_method_sil = results_df['silhouette'].idxmax()
best_method_ch = results_df['calinski_harabasz'].idxmax()

print(f"best method by silhouette score: {best_method_sil} ({results_df.loc[best_method_sil, 'silhouette']:.3f})")
print(f"best method by calinski-harabasz score: {best_method_ch} ({results_df.loc[best_method_ch, 'calinski_harabasz']:.3f})")

# ##Plot for evaluation scores comparison


# plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

results_df['silhouette'].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('silhouette score comparison')
ax1.set_ylabel('silhouette score')
ax1.tick_params(axis='x', rotation=45)

results_df['calinski_harabasz'].plot(kind='bar', ax=ax2, color='red')
ax2.set_title('calinski-harabasz score comparison')
ax2.set_ylabel('calinski-harabasz score')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("analysis complete!")
plot_clusters(pca_data, kmeans_labels, 'pca + k-means clustering', ['pc1', 'pc2'])
sil_pca_km, ch_pca_km = evaluate_clustering(pca_data, kmeans_labels, 'pca + k-means')
results['pca + k-means'] = {'silhouette': sil_pca_km, 'calinski_harabasz': ch_pca_km}
