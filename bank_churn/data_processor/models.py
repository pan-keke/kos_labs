from django.db import models

# Create your models here.

class Customer(models.Model):
    GEOGRAPHY_CHOICES = [
        ('France', 'France'),
        ('Spain', 'Spain'),
        ('Germany', 'Germany'),
    ]
    
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]

    customer_id = models.IntegerField(unique=True)
    surname = models.CharField(max_length=100)
    creditscore = models.IntegerField()
    geography = models.CharField(max_length=10, choices=GEOGRAPHY_CHOICES)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.IntegerField()
    tenure = models.IntegerField()
    balance = models.FloatField()
    numofproducts = models.IntegerField()
    hascrcard = models.BooleanField()
    isactivemember = models.BooleanField()
    estimatedsalary = models.FloatField()
    exited = models.BooleanField()
    
    class Meta:
        db_table = 'customers'
        
    def __str__(self):
        return f"Customer {self.customer_id} - {self.surname}"
