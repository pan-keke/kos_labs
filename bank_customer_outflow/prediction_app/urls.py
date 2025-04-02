from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('train/', views.train_model, name='train_model'),
]