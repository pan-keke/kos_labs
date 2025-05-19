from django.urls import path
from . import views


app_name = 'recommendation_system'

urlpatterns = [

    path('list/', views.RecommendationListView.as_view(), name='recommendation_list'),
    path('generate/<int:customer_id>/', views.generate_recommendations, name='generate_recommendations'),
] 