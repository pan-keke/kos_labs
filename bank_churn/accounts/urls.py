from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('activity-log/', views.UserActivityLogView.as_view(), name='activity_log'),
    path('user-management/', views.UserManagementView.as_view(), name='user_management'),
    path('user/<int:pk>/edit/', views.EditUserView.as_view(), name='edit_user'),
    path('user/<int:pk>/delete/', views.delete_user, name='delete_user'),
] 