import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


class ChurnModel:
    def __init__(self):
        self.model_path = os.path.join('prediction_app', 'services', 'churn_model.h5')
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = None

    def build_model(self, input_shape):
        """Build a neural network model for churn prediction"""
        model = Sequential([
            Dense(16, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=100, batch_size=32):
        """Train the model and save it"""
        # Get input shape from the data
        input_shape = X_train.shape[1]

        # Build model
        self.model = self.build_model(input_shape)

        # Prepare validation data
        validation_data = None
        if X_test is not None and y_test is not None:
            validation_data = (X_test, y_test)

        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )

        # Save model
        self.model.save(self.model_path)

        return history

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        # Get prediction probabilities
        y_pred = self.model.predict(X)

        return y_pred

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test)

        return {
            'loss': loss,
            'accuracy': accuracy
        }