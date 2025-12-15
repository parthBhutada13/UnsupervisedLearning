# Unsupervised Learning
# Mall Customer Segmentation using Unsupervised Learning

## Problem Statement

Retail environments generate large volumes of customer data, yet often lack explicit labels that define customer categories. Traditional rule-based segmentation fails to capture complex, non-linear relationships between customer attributes. This project addresses the challenge of discovering natural customer segments using unsupervised learning, enabling retailers to better understand customer behavior and optimize marketing and resource allocation strategies.

---

## Dataset Description

* **Dataset Name**: Mall Customers Dataset
* **Number of Records**: 200
* **Number of Features**: 5 (4 used for modeling)

### Original Features

* CustomerID (dropped)
* Gender (encoded)
* Age
* Annual Income (k$)
* Spending Score (1–100)

The dataset is clean, with no missing values, making it well-suited for clustering analysis.

## Results and Comparison

### Quantitative Evaluation

| Dimensionality Reduction + Clustering | Silhouette Score | Calinski–Harabasz Score |
| ------------------------------------- | ---------------- | ----------------------- |
| PCA + K-Means                         | 0.6880           | 514.15                  |
| t-SNE + K-Means                       | 0.7394           | 5750.68                 |
| PCA + DBSCAN                          | 0.6774           | 525.34                  |
| t-SNE + DBSCAN                        | **0.9075**       | **9015.30**             |
| PCA + Hierarchical                    | 0.6893           | 514.12                  |
| t-SNE + Hierarchical                  | 0.6888           | 7586.87                 |

### Best Performing Model

* **Best by Silhouette Score**: t-SNE + DBSCAN (0.908)
* **Best by Calinski–Harabasz Score**: t-SNE + DBSCAN (9015.30)

This combination consistently produced the most well-separated and interpretable clusters.

## Applications

* Retail customer segmentation
* Targeted marketing and promotions
* Customer lifetime value analysis

## Conclusion

This project demonstrates that combining dimensionality reduction with unsupervised clustering yields meaningful and actionable customer segments in retail analytics. Among all evaluated methods, t-SNE combined with DBSCAN achieved superior performance, highlighting the importance of non-linear representation learning for real-world customer data.

The proposed workflow is modular, reproducible, and extensible, making it suitable for academic projects, industry use cases, and further research.
