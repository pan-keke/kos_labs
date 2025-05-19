from django.db import models
from data_processor.models import Customer

# Create your models here.

class ChurnPrediction(models.Model):
    customer = models.ForeignKey(
        Customer,
        on_delete=models.CASCADE,
        related_name='churn_predictions'
    )
    prediction_date = models.DateTimeField(auto_now_add=True)
    churn_probability = models.FloatField()
    predicted_churn = models.BooleanField()
    
    class Meta:
        ordering = ['-prediction_date']
        
    def __str__(self):
        return f"{self.customer.surname} - {self.prediction_date.strftime('%Y-%m-%d %H:%M')}"
