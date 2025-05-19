"""
URL configuration for bank_churn project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth.decorators import login_required
from data_processor.views import CustomerListView

urlpatterns = [
    path('admin/', admin.site.urls),
    # Главная страница - список клиентов
    path('', login_required(CustomerListView.as_view()), name='home'),
    # Аутентификация и управление пользователями
    path('accounts/', include('accounts.urls')),
    # Обработка данных и основные функции
    path('data/', include('data_processor.urls')),
    # Прогнозирование оттока
    path('churn/', include('churn_predictor.urls')),
    # Система рекомендаций
    path('recommendations/', include('recommendation_system.urls')),
]
