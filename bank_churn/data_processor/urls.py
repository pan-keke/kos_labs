from django.urls import path
from . import views

app_name = 'data_processor'

urlpatterns = [
    path('', views.CustomerListView.as_view(), name='customer_list'),
    path('customer/<int:pk>/', views.CustomerDetailView.as_view(), name='customer_detail'),
    path('predict/', views.ChurnPredictionView.as_view(), name='input_form'),
    path('predict/result/', views.PredictionResultView.as_view(), name='prediction_result'),
] 