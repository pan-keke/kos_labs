from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, TemplateView, FormView
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from django.contrib import messages
from data_processor.models import Customer
from .models import ChurnPrediction
from .utils import prepare_data, train_model, predict_churn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import os
from datetime import datetime
import threading
import time

# Create your views here.

# Глобальная переменная для хранения прогресса обучения
training_progress = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'status': 'idle'
}

class ChurnPredictionListView(LoginRequiredMixin, ListView):
    model = ChurnPrediction
    template_name = 'churn_predictor/prediction_list.html'
    context_object_name = 'predictions'
    paginate_by = 50

class PredictionHistoryView(LoginRequiredMixin, ListView):
    model = ChurnPrediction
    template_name = 'churn_predictor/prediction_history.html'
    context_object_name = 'predictions'
    paginate_by = 20
    ordering = ['-prediction_date']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['total_predictions'] = ChurnPrediction.objects.count()
        context['high_risk_count'] = ChurnPrediction.objects.filter(predicted_churn=True).count()
        context['low_risk_count'] = ChurnPrediction.objects.filter(predicted_churn=False).count()
        return context

def get_training_status(request):
    """API endpoint для получения статуса обучения"""
    return JsonResponse(training_progress)

class ModelTrainingView(UserPassesTestMixin, TemplateView):
    template_name = 'churn_predictor/train_model.html'
    
    def test_func(self):
        return self.request.user.is_superuser
    
    def post(self, request, *args, **kwargs):
        try:
            # Получаем параметры обучения из формы
            epochs = int(request.POST.get('epochs', 100))
            batch_size = int(request.POST.get('batch_size', 32))
            learning_rate = float(request.POST.get('learning_rate', 0.001))
            validation_split = float(request.POST.get('validation_split', 0.2))
            
            # Проверяем валидность параметров
            if not (1 <= epochs <= 1000):
                raise ValueError("Количество эпох должно быть от 1 до 1000")
            if batch_size not in [16, 32, 64, 128]:
                raise ValueError("Недопустимый размер батча")
            if not (0.0001 <= learning_rate <= 0.1):
                raise ValueError("Скорость обучения должна быть от 0.0001 до 0.1")
            if not (0.1 <= validation_split <= 0.3):
                raise ValueError("Доля валидационной выборки должна быть от 0.1 до 0.3")
            
            # Получаем все данные клиентов из базы данных
            customers = Customer.objects.all().values(
                'creditscore', 'geography', 'gender', 'age', 'tenure',
                'balance', 'numofproducts', 'hascrcard',
                'isactivemember', 'estimatedsalary', 'exited'
            )
            
            # Преобразуем в pandas DataFrame
            data = pd.DataFrame(list(customers))
            
            # Проверяем, есть ли данные для обучения
            if len(data) == 0:
                messages.error(request, "Нет данных для обучения. Пожалуйста, загрузите данные клиентов.")
                return self.get(request, *args, **kwargs)
            
            # Обучаем модель с заданными параметрами
            training_results = train_model(
                data=data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split
            )
            
            # Преобразуем numpy значения в обычные Python типы
            metrics = {}
            for key, value in training_results['metrics'].items():
                if isinstance(value, (np.float32, np.float64)):
                    metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics[key] = float(value.item())
                else:
                    metrics[key] = value

            # Ensure we have all required metrics
            if 'accuracy' not in metrics:
                metrics['accuracy'] = metrics.get('val_accuracy', 0.0)
            if 'auc' not in metrics:
                metrics['auc'] = metrics.get('val_auc', 0.0)
            if 'loss' not in metrics:
                metrics['loss'] = metrics.get('val_loss', 0.0)

            # Round metrics to 4 decimal places
            metrics = {k: round(float(v), 4) for k, v in metrics.items()}

            # Сохраняем результаты в сессии
            request.session['training_results'] = {
                'metrics': metrics,
                'training_params': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'validation_split': validation_split
                }
            }
            
            messages.success(request, "Модель успешно обучена!")
            
        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"Ошибка при обучении модели: {str(e)}")
        
        return self.get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        model_path = 'models/churn_model.h5'
        scaler_path = 'models/scaler.pkl'
        
        context['model_exists'] = os.path.exists(model_path)
        context['scaler_exists'] = os.path.exists(scaler_path)
        if context['model_exists']:
            context['last_modified'] = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        # Добавляем информацию о количестве данных
        context['total_customers'] = Customer.objects.count()
        
        # Добавляем результаты обучения из сессии
        if 'training_results' in self.request.session:
            context['training_results'] = self.request.session['training_results']
            del self.request.session['training_results']
        
        return context

def predict_churn(request, customer_id):
    customer = get_object_or_404(Customer, id=customer_id)
    
    # Создаем словарь с данными клиента
    customer_data = {
        'creditscore': customer.creditscore,
        'geography': customer.geography,
        'gender': customer.gender,
        'age': customer.age,
        'tenure': customer.tenure,
        'balance': customer.balance,
        'numofproducts': customer.numofproducts,
        'hascrcard': int(customer.hascrcard),
        'isactivemember': int(customer.isactivemember),
        'estimatedsalary': customer.estimatedsalary
    }
    
    try:
        # Используем функцию predict_churn из utils.py
        from .utils import predict_churn as predict_churn_util
        result = predict_churn_util(customer_data)
        
        # Если нет ошибки, сохраняем предсказание
        if not result.get('error'):
            ChurnPrediction.objects.create(
                customer=customer,
                churn_probability=result['probability'],
                predicted_churn=result['will_churn']
            )
        
        return JsonResponse({
            'success': not bool(result.get('error')),
            'probability': result.get('probability'),
            'will_churn': result.get('will_churn'),
            'error': result.get('error')
        })
        
    except Exception as e:
        import traceback
        print(f"Error in predict_churn view: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
