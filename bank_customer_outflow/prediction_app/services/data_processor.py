import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import os
import json


class DataProcessor:
    def __init__(self):
        self.label_country_encoder = LabelEncoder()
        self.label_gender_encoder = LabelEncoder()
        self.column_transformer = None
        self.scaler = StandardScaler()
        self.initialize_encoders()

    def initialize_encoders(self):
        """Initialize encoders with available data"""
        # Load initial data to fit encoders
        data_path = os.path.join('data', 'Churn_Modelling.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

            # Convert HasCrCard and IsActiveMember to integers explicitly
            data['HasCrCard'] = data['HasCrCard'].astype(int)
            data['IsActiveMember'] = data['IsActiveMember'].astype(int)

            # Fit country encoder
            self.label_country_encoder.fit(data['Geography'])

            # Fit gender encoder
            self.label_gender_encoder.fit(data['Gender'])

            # Extract features for column transformer
            X = data.iloc[:, 0:10].values
            X[:, 1] = self.label_country_encoder.transform(X[:, 1])
            X[:, 2] = self.label_gender_encoder.transform(X[:, 2])

            # Initialize column transformer - note we're only transforming the Geography column (index 1)
            self.column_transformer = ColumnTransformer(
                [("countries", OneHotEncoder(), [1])],
                remainder="passthrough"
            )

            # Fit column transformer
            X_transformed = self.column_transformer.fit_transform(X)

            # Fit scaler
            self.scaler.fit(X_transformed)

            # Save encoder mappings
            self.save_mappings()
        else:
            # Fallback for when the file doesn't exist
            self.label_country_encoder.fit(['France', 'Spain', 'Germany'])
            self.label_gender_encoder.fit(['Male', 'Female'])
            self.column_transformer = ColumnTransformer(
                [("countries", OneHotEncoder(), [1])],
                remainder="passthrough"
            )

    def save_mappings(self):
        """Save encoder mappings for future reference"""
        mappings = {
            'countries': list(self.label_country_encoder.classes_),
            'genders': list(self.label_gender_encoder.classes_)
        }

        with open(os.path.join('prediction_app', 'services', 'encoder_mappings.json'), 'w') as f:
            json.dump(mappings, f)

    def load_data(self, filepath):
        """Load and preprocess training data"""
        data = pd.read_csv(filepath)
        data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

        X = data.iloc[:, 0:10].values
        y = data.iloc[:, 10].values

        # Transform categorical variables
        X[:, 1] = self.label_country_encoder.transform(X[:, 1])
        X[:, 2] = self.label_gender_encoder.transform(X[:, 2])

        # Apply one-hot encoding and scaling
        X = self.column_transformer.transform(X)
        X = self.scaler.transform(X)

        return X, y

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Ensure all numeric values are properly typed
        try:
            # Convert input to correct format
            input_array = np.array([
                [
                    int(input_data['credit_score']),
                    input_data['geography'],
                    input_data['gender'],
                    int(input_data['age']),
                    int(input_data['tenure']),
                    float(input_data['balance']),
                    int(input_data['num_of_products']),
                    int(input_data['has_cr_card']),
                    int(input_data['is_active_member']),
                    float(input_data['estimated_salary'])
                ]
            ], dtype=object)  # Use object dtype to handle mixed types

            # Print for debugging
            print("Input array types before transform:", [type(x) for x in input_array[0]])

            # Validate that geography and gender values are in the trained categories
            if input_data['geography'] not in self.label_country_encoder.classes_:
                raise ValueError(
                    f"Unknown geography: {input_data['geography']}. Expected one of {list(self.label_country_encoder.classes_)}")

            if input_data['gender'] not in self.label_gender_encoder.classes_:
                raise ValueError(
                    f"Unknown gender: {input_data['gender']}. Expected one of {list(self.label_gender_encoder.classes_)}")

            # Apply categorical transformations
            input_array[:, 1] = self.label_country_encoder.transform([input_data['geography']])
            input_array[:, 2] = self.label_gender_encoder.transform([input_data['gender']])

            # Print for debugging
            print("Input array after label encoding:", input_array)

            # Apply one-hot encoding and scaling
            input_transformed = self.column_transformer.transform(input_array)
            input_scaled = self.scaler.transform(input_transformed)

            return input_scaled

        except Exception as e:
            print(f"Error in preprocess_input: {e}")
            print(f"Input data: {input_data}")
            raise