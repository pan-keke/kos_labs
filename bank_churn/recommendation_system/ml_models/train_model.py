import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def train_clustering_model(csv_file, n_clusters=5):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Select features for clustering
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    print(f'Silhouette Score: {silhouette_avg:.4f}')
    
    # Analyze clusters
    df['Cluster'] = kmeans.labels_
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f'\nCluster {i} statistics:')
        print(cluster_data[features].describe().round(2))
    
    # Save model and scaler
    joblib.dump(kmeans, 'kmeans_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return kmeans, scaler

if __name__ == '__main__':
    train_clustering_model('../../bank.csv') 