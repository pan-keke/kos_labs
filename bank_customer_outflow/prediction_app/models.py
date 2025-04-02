from django.db import models


class ChurnPrediction(models.Model):
    credit_score = models.IntegerField()
    geography = models.CharField(max_length=50)
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    tenure = models.IntegerField()
    balance = models.FloatField()
    num_of_products = models.IntegerField()
    has_cr_card = models.BooleanField()
    is_active_member = models.BooleanField()
    estimated_salary = models.FloatField()
    prediction_result = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for client (ID: {self.id})"