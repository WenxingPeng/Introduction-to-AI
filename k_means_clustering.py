import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# construct the 2-D data set
X = np.array([[1, 1], [1, 5], [2, 1], [3, 4], [4, 5], [10, 8], [7, 9], [1, 3], [2, 2], [4, 2.5], [8, 5], [7, 7], [5, 6], [9, 6], [9, 4], [4, 9], [6, 8], [10, 4]])
print(X.shape)
# plot the data set
plt.scatter(X[:, 0],X[:, 1], label='True Position')

# apply KMeans with K = 3
kmeans = KMeans(n_clusters=3)  # n_clusters: number of clusters
kmeans.fit(X)

# print cluster centers and label each data point to corresponding center
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# plot the clusters and cluster centroids
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')

from scipy.spatial.distance import cdist, euclidean
import numpy as np
distortions = []
K = range(1, 10)
for k in K:
    kmeanmodel = KMeans(n_clusters=k).fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanmodel.cluster_centers_, 'euclidean')**2, axis=1)/X.shape[0]))

plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method Showing the optimal k')
plt.show()







