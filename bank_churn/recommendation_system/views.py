from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView
from django.http import JsonResponse
from data_processor.models import Customer
from .models import CustomerCluster, CustomerSegment, Recommendation
import os

class RecommendationListView(ListView):
    model = Recommendation
    template_name = 'recommendation_system/recommendation_list.html'
    context_object_name = 'recommendations'
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add priority counts to context
        context['high_priority_count'] = Recommendation.objects.filter(priority=1).count()
        context['medium_priority_count'] = Recommendation.objects.filter(priority=2).count()
        context['low_priority_count'] = Recommendation.objects.filter(priority=3).count()
        return context

def generate_recommendations(request, customer_id):
    customer = get_object_or_404(Customer, id=customer_id)
    
    try:
        # Generate basic recommendations based on customer data
        recommendations = []
        
        # High-value customers
        if customer.balance > 100000 or customer.estimatedsalary > 150000:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Premium Banking Services',
                    description='Consider our premium banking services with exclusive benefits and personalized support.',
                    priority=1
                )
            )
        
        # Customers with no credit card
        if not customer.hascrcard:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Credit Card Offer',
                    description='Explore our credit card options with special rewards and benefits.',
                    priority=2
                )
            )
        
        # Inactive customers
        if not customer.isactivemember:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Account Activity Benefits',
                    description='Discover the benefits of active account usage and special promotions.',
                    priority=1
                )
            )
        
        # Single product customers
        if customer.numofproducts == 1:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Product Diversification',
                    description='Explore additional banking products to better serve your financial needs.',
                    priority=3
                )
            )
        
        # Low balance customers
        if customer.balance < 10000:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Savings Opportunities',
                    description='Learn about our savings accounts and investment options to grow your wealth.',
                    priority=2
                )
            )
        
        return JsonResponse({
            'success': True,
            'recommendations': [
                {
                    'title': rec.title,
                    'description': rec.description,
                    'priority': rec.priority
                }
                for rec in recommendations
            ]
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
