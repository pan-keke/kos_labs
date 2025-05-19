from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    is_analyst = models.BooleanField(default=False)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        db_table = 'users'
        
    def __str__(self):
        return f"{self.username} ({'Analyst' if self.is_analyst else 'Admin'})"

class UserActivity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=50)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'user_activities'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.user.username} - {self.activity_type} at {self.created_at}"
