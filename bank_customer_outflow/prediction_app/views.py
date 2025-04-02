from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.model_selection import train_test_split
import os
import json

from .models import ChurnPrediction
from .services.data_processor import DataProcessor
from .services.ml_model import ChurnModel

# Initialize data processor and model
data_processor = DataProcessor()
churn_model = ChurnModel()


def index(request):
    """Render the main prediction form"""
    # Check if model exists, if not, train it first
    model_path = os.path.join('prediction_app', 'services', 'churn_model.h5')
    if not os.path.exists(model_path):
        # Model not trained yet, show message
        return render(request, 'index.html', {'model_ready': False})

    # Get encoder mappings for countries and genders
    mappings_path = os.path.join('prediction_app', 'services', 'encoder_mappings.json')

    if os.path.exists(mappings_path):
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
            countries = mappings['countries']
            genders = mappings['genders']
    else:
        countries = ['France', 'Spain', 'Germany']
        genders = ['Male', 'Female']

    return render(request, 'index.html', {
        'model_ready': True,
        'countries': countries,
        'genders': genders
    })


def train_model(request):
    """Train the model with the dataset"""
    # Load and preprocess data
    data_path = os.path.join('data', 'Churn_Modelling.csv')

    if not os.path.exists(data_path):
        return JsonResponse({'error': 'Training data not found'}, status=404)

    # Load and preprocess data
    X, y = data_processor.load_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    history = churn_model.train(X_train, y_train, X_test, y_test)

    # Evaluate model
    evaluation = churn_model.evaluate(X_test, y_test)

    return JsonResponse({
        'status': 'success',
        'accuracy': evaluation['accuracy']
    })


def predict(request):
    """Handle prediction form submission"""
    if request.method == 'POST':
        try:
            # Extract input data and ensure proper types
            has_cr_card_val = request.POST.get('has_cr_card')
            is_active_member_val = request.POST.get('is_active_member')

            # Convert checkbox values properly
            has_cr_card_int = 1 if has_cr_card_val == 'on' else 0
            is_active_member_int = 1 if is_active_member_val == 'on' else 0

            input_data = {
                'credit_score': int(request.POST.get('credit_score')),
                'geography': request.POST.get('geography'),
                'gender': request.POST.get('gender'),
                'age': int(request.POST.get('age')),
                'tenure': int(request.POST.get('tenure')),
                'balance': float(request.POST.get('balance')),
                'num_of_products': int(request.POST.get('num_of_products')),
                'has_cr_card': has_cr_card_int,
                'is_active_member': is_active_member_int,
                'estimated_salary': float(request.POST.get('estimated_salary'))
            }

            # Preprocess input
            X = data_processor.preprocess_input(input_data)

            # Make prediction
            prediction = float(churn_model.predict(X)[0][0])
            prediction_percent = prediction * 100  # Calculate percentage here

            # Save prediction to database
            churn_prediction = ChurnPrediction(
                credit_score=input_data['credit_score'],
                geography=input_data['geography'],
                gender=input_data['gender'],
                age=input_data['age'],
                tenure=input_data['tenure'],
                balance=input_data['balance'],
                num_of_products=input_data['num_of_products'],
                has_cr_card=input_data['has_cr_card'],
                is_active_member=input_data['is_active_member'],
                estimated_salary=input_data['estimated_salary'],
                prediction_result=prediction
            )
            churn_prediction.save()

            # Determine churn status
            if prediction > 0.5:
                churn_status = 'Вероятно уйдет'
            else:
                churn_status = 'Вероятно не уйдет'

            # Return result
            return render(request, 'result.html', {
                'prediction': prediction,
                'prediction_percent': prediction_percent,  # Pass percentage to template
                'churn_status': churn_status,
                'input_data': input_data
            })

        except Exception as e:
            # Handle errors gracefully
            print(f"Error in prediction: {e}")
            # Get encoder mappings for countries and genders
            mappings_path = os.path.join('prediction_app', 'services', 'encoder_mappings.json')

            if os.path.exists(mappings_path):
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    countries = mappings['countries']
                    genders = mappings['genders']
            else:
                countries = ['France', 'Spain', 'Germany']
                genders = ['Male', 'Female']

            return render(request, 'index.html', {
                'error_message': f"Error processing your request: {e}",
                'model_ready': True,
                'countries': countries,
                'genders': genders
            })

    # If not POST method, redirect to index
    return redirect('index')


@api_view(['POST'])
def predict_api(request):
    """API endpoint for predictions"""
    data = request.data

    # Validate input data
    required_fields = [
        'credit_score', 'geography', 'gender', 'age', 'tenure',
        'balance', 'num_of_products', 'has_cr_card', 'is_active_member', 'estimated_salary'
    ]

    for field in required_fields:
        if field not in data:
            return Response({'error': f'Missing required field: {field}'}, status=400)

    # Process input data
    input_data = {
        'credit_score': int(data['credit_score']),
        'geography': data['geography'],
        'gender': data['gender'],
        'age': int(data['age']),
        'tenure': int(data['tenure']),
        'balance': float(data['balance']),
        'num_of_products': int(data['num_of_products']),
        'has_cr_card': 1 if bool(data['has_cr_card']) else 0,  # Convert to integer
        'is_active_member': 1 if bool(data['is_active_member']) else 0,  # Convert to integer
        'estimated_salary': float(data['estimated_salary'])
    }

    # Preprocess input
    X = data_processor.preprocess_input(input_data)

    # Make prediction
    prediction = float(churn_model.predict(X)[0][0])

    # Save prediction to database
    churn_prediction = ChurnPrediction(
        credit_score=input_data['credit_score'],
        geography=input_data['geography'],
        gender=input_data['gender'],
        age=input_data['age'],
        tenure=input_data['tenure'],
        balance=input_data['balance'],
        num_of_products=input_data['num_of_products'],
        has_cr_card=input_data['has_cr_card'],
        is_active_member=input_data['is_active_member'],
        estimated_salary=input_data['estimated_salary'],
        prediction_result=prediction
    )
    churn_prediction.save()

    # Determine churn status
    if prediction > 0.5:
        churn_status = 'Вероятно уйдет'
    else:
        churn_status = 'Вероятно не уйдет'

    # Return prediction result
    return Response({
        'prediction': prediction,
        'churn_status': churn_status
    })