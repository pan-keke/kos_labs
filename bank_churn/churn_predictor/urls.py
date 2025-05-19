from django.urls import path
from . import views

app_name = 'churn_predictor'

urlpatterns = [
    path('', views.ChurnPredictionListView.as_view(), name='prediction_list'),
    path('predict/<int:customer_id>/', views.predict_churn, name='predict'),
    path('history/', views.PredictionHistoryView.as_view(), name='prediction_history'),
    path('train/', views.ModelTrainingView.as_view(), name='train_model'),
    path('train/status/', views.get_training_status, name='training_status'),
] 