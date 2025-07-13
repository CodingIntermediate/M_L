from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the data
iris = load_iris()
X = iris.data

# Step 2: Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Step 3: Show the cluster labels
print("Predicted Cluster Labels:", kmeans.labels_)

# Step 4: Visualize the clusters (2 features only)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering (Iris Dataset)')
plt.show()
# (graph)