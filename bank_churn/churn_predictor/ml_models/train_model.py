import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_churn_model(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Prepare features
    X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary']]
    y = df['Exited']
    
    # Convert categorical variables
    X = pd.get_dummies(X, columns=['Geography', 'Gender'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'churn_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Print model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f'Train accuracy: {train_score:.4f}')
    print(f'Test accuracy: {test_score:.4f}')
    
    return model, scaler

if __name__ == '__main__':
    train_churn_model('../../bank.csv') 