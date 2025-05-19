import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import joblib
import os

def prepare_data(data):
    """
    Prepare data for model training or prediction
    """
    # For training, data is a DataFrame. For prediction, it's a dict
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Select features for model training (excluding RowNumber, CustomerId, Surname)
    feature_columns = [
        'creditscore', 'geography', 'gender', 'age', 'tenure',
        'balance', 'numofproducts', 'hascrcard',
        'isactivemember', 'estimatedsalary'
    ]
    
    if 'exited' in data.columns:
        y = data['exited'].values
    else:
        y = None
    
    X = data[feature_columns].copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['geography', 'gender']
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Convert to numpy array
    X = X.values
    
    return X, y, label_encoders

def create_model(learning_rate=0.001):
    """
    Create neural network model for churn prediction
    """
    model = Sequential([
        # First layer
        Dense(128, activation='relu', input_shape=(10,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second layer
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third layer
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_model(data, epochs=100, batch_size=32, learning_rate=0.001, validation_split=0.2, progress_callback=None):
    """
    Train the neural network model
    """
    # Prepare data
    X, y, _ = prepare_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = create_model(learning_rate=learning_rate)
    
    # Class weights for imbalanced data
    class_weights = dict(zip(
        np.unique(y_train),
        1 / np.bincount(y_train) * len(y_train) / 2
    ))
    
    # Create model directory if it doesn't exist
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Custom callback для отслеживания прогресса
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, epochs)
    
    # Callbacks
    callbacks = [
        ProgressCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=40,
            restore_best_weights=True,
            min_delta=0.0001,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=20,
            min_lr=0.00001
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model on test set
    test_metrics = model.evaluate(X_test_scaled, y_test, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, test_metrics))
    
    # Add validation metrics from last epoch
    for key in ['loss', 'accuracy', 'auc']:
        val_key = f'val_{key}'
        if val_key in history.history:
            metrics_dict[val_key] = float(history.history[val_key][-1])
    
    # Save model and scaler
    model.save(os.path.join(model_dir, 'churn_model.h5'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    return {
        'metrics': metrics_dict,
        'history': history.history
    }

def predict_churn(data):
    """
    Make churn prediction for a single customer
    """
    try:
        # Load model and scaler
        model_dir = os.path.join('models')
        model_path = os.path.join(model_dir, 'churn_model.h5')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Файлы модели не найдены. Пожалуйста, сначала обучите модель.")
        
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Prepare input data
        X, _, _ = prepare_data(data)
        if X is None or len(X) == 0:
            raise ValueError("Не удалось подготовить входные данные")
            
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        probability = float(model.predict(X_scaled, verbose=0)[0][0])
        
        return {
            'probability': probability * 100,  # Convert to percentage
            'will_churn': probability > 0.5
        }
        
    except Exception as e:
        import traceback
        print(f"Error in predict_churn: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'probability': None,
            'will_churn': None
        }

def get_feature_importance():
    """
    Получение важности признаков для объяснения предсказания
    """
    # Здесь можно добавить логику анализа важности признаков
    # Например, используя SHAP values или другие методы
    pass 