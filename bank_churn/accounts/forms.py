from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User

class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'is_analyst')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['is_analyst'].label = "Register as Analyst"
        self.fields['is_analyst'].help_text = "Check this if you are registering as an analyst"
        
class UserLoginForm(AuthenticationForm):
    class Meta:
        model = User
        fields = ('username', 'password')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control'})
        self.fields['password'].widget.attrs.update({'class': 'form-control'})

    def get_user(self):
        return User.objects.get(username=self.cleaned_data['username'])

class UserEditForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'is_active', 'is_analyst']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_analyst': forms.CheckboxInput(attrs={'class': 'form-check-input'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['is_analyst'].label = "Analyst Role"
        self.fields['is_analyst'].help_text = "Check this if the user should have analyst privileges" 