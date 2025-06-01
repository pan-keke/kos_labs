from django.shortcuts import render
from django.views.generic import ListView, DetailView, FormView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from .models import Customer
from churn_predictor.utils import predict_churn
from django import forms
from django.views import View
from django.shortcuts import render


class ChurnPredictionForm(forms.Form):

    creditscore = forms.IntegerField(
        min_value=300, max_value=850,
        help_text="Credit score (300-850)",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[0-9]*',
            'inputmode': 'numeric',
            'placeholder': '300-850'
        })
    )
    geography = forms.ChoiceField(
        choices=Customer.GEOGRAPHY_CHOICES,
        help_text="Customer's country",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    gender = forms.ChoiceField(
        choices=Customer.GENDER_CHOICES,
        help_text="Customer's gender",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    age = forms.IntegerField(
        min_value=18, max_value=100,
        help_text="Customer's age (18-100)",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[0-9]*',
            'inputmode': 'numeric',
            'placeholder': '18-100'
        })
    )
    tenure = forms.IntegerField(
        min_value=0, max_value=50,
        help_text="Years with bank (0-50)",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[0-9]*',
            'inputmode': 'numeric',
            'placeholder': '0-50'
        })
    )
    balance = forms.FloatField(
        min_value=0,
        help_text="Account balance",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'inputmode': 'decimal',
            'placeholder': 'Enter balance',
            'step': 'any'
        })
    )
    numofproducts = forms.IntegerField(
        min_value=1, max_value=4,
        help_text="Number of bank products (1-4)",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[1-4]',
            'inputmode': 'numeric',
            'placeholder': '1-4'
        })
    )
    hascrcard = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Has a credit card?",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    isactivemember = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Is an active member?",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    estimatedsalary = forms.FloatField(
        min_value=0,
        help_text="Estimated annual salary",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'inputmode': 'decimal',
            'placeholder': 'Enter salary',
            'step': 'any'
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        # Convert boolean fields to integers for the model
        cleaned_data['hascrcard'] = 1 if cleaned_data.get('hascrcard') else 0
        cleaned_data['isactivemember'] = 1 if cleaned_data.get('isactivemember') else 0
        return cleaned_data


class CustomerListView(ListView):
    model = Customer
    template_name = 'data_processor/customer_list.html'
    context_object_name = 'customers'
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['show_loader'] = True
        return context

class CustomerDetailView(DetailView):
    model = Customer
    template_name = 'data_processor/customer_detail.html'
    context_object_name = 'customer'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        customer = self.get_object()
        context['churn_predictions'] = customer.churn_predictions.all()
        context['recommendations'] = customer.recommendations.all()
        return context


class ChurnPredictionView(LoginRequiredMixin, FormView):
    template_name = 'data_processor/input_form.html'
    form_class = ChurnPredictionForm
    success_url = reverse_lazy('data_processor:prediction_result')

    def form_valid(self, form):
        # Получаем данные из формы
        data = form.cleaned_data

        # Предсказываем вероятность оттока
        prediction_result = predict_churn(data)

        if prediction_result.get('error'):
            # Если произошла ошибка, показываем её пользователю
            form.add_error(None, prediction_result['error'])
            return self.form_invalid(form)

        # Сохраняем результат в сессии для отображения
        self.request.session['prediction_result'] = {
            'probability': prediction_result['probability'],
            'will_churn': prediction_result['will_churn'],
            'customer_data': data
        }

        return super().form_valid(form)


class PredictionResultView(LoginRequiredMixin, TemplateView):
    template_name = 'data_processor/prediction_result.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['prediction'] = self.request.session.get('prediction_result')
        return context


class CustomerDetailView(DetailView):
    model = Customer
    template_name = 'data_processor/customer_detail.html'
    context_object_name = 'customer'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        customer = self.get_object()

        # Получаем GET-параметр
        filter_param = self.request.GET.get('filter')

        # Все предсказания
        churn_predictions = customer.churn_predictions.all()

        # Фильтрация
        if filter_param == 'churn':
            churn_predictions = churn_predictions.filter(predicted_churn=True)
        elif filter_param == 'stay':
            churn_predictions = churn_predictions.filter(predicted_churn=False)

        context['churn_predictions'] = churn_predictions
        context['recommendations'] = customer.recommendations.all()

        # Общая статистика (для отображения в карточках)
        total = customer.churn_predictions.count()
        churn_count = customer.churn_predictions.filter(predicted_churn=True).count()
        stay_count = total - churn_count

        context['total_churn_predictions'] = total
        context['churn_count'] = churn_count
        context['stay_count'] = stay_count

        return context


