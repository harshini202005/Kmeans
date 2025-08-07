from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import pickle

X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])


kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Feature1', 'Feature2']])


with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)


df.to_csv("cluster_data.csv", index=False)
