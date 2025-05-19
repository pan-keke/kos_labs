from django.db import models
from data_processor.models import Customer

# Create your models here.

class CustomerCluster(models.Model):
    cluster_id = models.IntegerField()
    cluster_name = models.CharField(max_length=100)
    cluster_description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'customer_clusters'
        
    def __str__(self):
        return f"Cluster {self.cluster_id}: {self.cluster_name}"

class CustomerSegment(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='segments')
    cluster = models.ForeignKey(CustomerCluster, on_delete=models.CASCADE, related_name='customers')
    assigned_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'customer_segments'
        
    def __str__(self):
        return f"{self.customer.surname} in {self.cluster.cluster_name}"

class Recommendation(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='recommendations')
    cluster = models.ForeignKey(CustomerCluster, on_delete=models.CASCADE, related_name='recommendations', null=True, blank=True)
    title = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    priority = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'recommendations'
        ordering = ['-priority', '-created_at']
        
    def __str__(self):
        return f"Recommendation for {self.customer.surname}: {self.title}"
