import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def train_clustering_model(data_path, n_clusters=5):
    # Load data
    df = pd.read_csv(data_path)
    
    # Select features for clustering
    features = [
        'CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary'
    ]
    
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Save model and scaler
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    print("Model and scaler saved successfully!")
    return kmeans, scaler

if __name__ == '__main__':
    # Assuming the CSV file is in the project root directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'bank.csv')
    if os.path.exists(data_path):
        train_clustering_model(data_path)
    else:
        print(f"Error: Could not find data file at {data_path}") 