import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import time 

# K-Means Function
def kmeans_dm(input_data, number_of_clusters):
    start_time = time.time() 
    
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42
    )
    
    labels = model.fit_predict(input_data)
    
    elapsed_time = time.time() - start_time 
    
    return labels, elapsed_time

# K-Medoids Function
def kmedoids_dm(input_data, number_of_clusters):
    start_time = time.time()  
    
    model = KMedoids(
        n_clusters=number_of_clusters,
        random_state=42
    )
    
    labels = model.fit_predict(input_data)
    
    elapsed_time = time.time() - start_time  
    
    return labels, elapsed_time

if __name__ == "__main__":
    
    
    input_file = sys.argv[1]
    number_of_clusters = int(sys.argv[2])
    
    
    data = pd.read_csv(input_file)
    
    print(f"\n{'='*60}")
    print(f"Clustering with k = {number_of_clusters}")
    print(f"{'='*60}\n")
    
    #  K-Means
    kmeans_labels, kmeans_time = kmeans_dm(data, number_of_clusters)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)
    
    #  K-Medoids
    kmedoids_labels, kmedoids_time = kmedoids_dm(data, number_of_clusters)
    kmedoids_silhouette = silhouette_score(data, kmedoids_labels)
    
    #  Results
    print("K-Means Results:")
    print(f"  - Silhouette Coefficient: {kmeans_silhouette:.4f}")
    print(f"  - Running Time: {kmeans_time:.4f} seconds")
    print(f"  - Cluster Labels (first 20): {kmeans_labels[:20]}")
    
    print("\nK-Medoids Results:")
    print(f"  - Silhouette Coefficient: {kmedoids_silhouette:.4f}")
    print(f"  - Running Time: {kmedoids_time:.4f} seconds")
    print(f"  - Cluster Labels (first 20): {kmedoids_labels[:20]}")
    
    print(f"\n{'='*60}\n")