from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView
from django.http import JsonResponse
from data_processor.models import Customer
from .models import CustomerCluster, CustomerSegment, Recommendation
import os
from django.shortcuts import render
from .models import Recommendation  # если используется модель Recommendation

class RecommendationListView(ListView):
    model = Recommendation
    template_name = 'recommendation_system/recommendation_list.html'
    context_object_name = 'recommendations'
    paginate_by = 50

    def get_queryset(self):
        queryset = super().get_queryset()
        filter_type = self.request.GET.get('filter')
        
        if filter_type == 'high':
            queryset = queryset.filter(priority=1)
        elif filter_type == 'medium':
            queryset = queryset.filter(priority=2)
        elif filter_type == 'low':
            queryset = queryset.filter(priority=3)
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add priority counts to context
        context['high_priority_count'] = Recommendation.objects.filter(priority=1).count()
        context['medium_priority_count'] = Recommendation.objects.filter(priority=2).count()
        context['low_priority_count'] = Recommendation.objects.filter(priority=3).count()
        context['current_filter'] = self.request.GET.get('filter', '')
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
                    title='Премиальное банковское обслуживание',
                    description='Воспользуйтесь нашими премиальными банковскими услугами с эксклюзивными преимуществами и персональной поддержкой.',
                    priority=1
                )
            )
        
        # Customers with no credit card
        if not customer.hascrcard:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Предложение кредитной карты',
                    description='Ознакомьтесь с нашими кредитными картами со специальными вознаграждениями и преимуществами.',
                    priority=2
                )
            )
        
        # Inactive customers
        if not customer.isactivemember:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Преимущества активности счета',
                    description='Узнайте о преимуществах активного использования счета и специальных акциях.',
                    priority=1
                )
            )
        
        # Single product customers
        if customer.numofproducts == 1:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Диверсификация продуктов',
                    description='Изучите дополнительные банковские продукты для лучшего обслуживания ваших финансовых потребностей.',
                    priority=3
                )
            )
        
        # Low balance customers
        if customer.balance < 10000:
            recommendations.append(
                Recommendation.objects.create(
                    customer=customer,
                    title='Возможности для сбережений',
                    description='Узнайте о наших сберегательных счетах и инвестиционных возможностях для роста вашего капитала.',
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
