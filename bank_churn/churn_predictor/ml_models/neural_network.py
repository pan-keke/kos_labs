import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(df):
    # Select features
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    # Create dummy variables for categorical features
    geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')
    gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
    
    # Combine all features
    X = pd.concat([df[features], geography_dummies, gender_dummies], axis=1)
    y = df['Exited']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_neural_network(csv_file, epochs=50, batch_size=32):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Prepare data
    X_scaled, y, scaler = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_model()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    model.save('neural_network_model')
    np.save('feature_scaler.npy', scaler)
    
    return model, scaler, history

def predict_churn(model, scaler, customer_data):
    # Prepare customer data
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Geography_France', 'Geography_Germany', 'Geography_Spain',
                'Gender_Female', 'Gender_Male']
    
    # Scale features
    scaled_data = scaler.transform(customer_data[features])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    return prediction[0][0]

if __name__ == '__main__':
    train_neural_network('../../bank.csv') 