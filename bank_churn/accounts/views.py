from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages
from django.views.generic import ListView, UpdateView
from django.utils.decorators import method_decorator
from django.urls import reverse_lazy
from .forms import UserRegistrationForm, UserLoginForm, UserEditForm
from .models import User, UserActivity
from django.contrib.admin.views.decorators import staff_member_required

def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserActivity.objects.create(
                user=user,
                activity_type='REGISTRATION',
                description='User registered',
                ip_address=request.META.get('REMOTE_ADDR')
            )
            login(request, user)
            return redirect('home')
    else:
        form = UserRegistrationForm()
    return render(request, 'accounts/register.html', {'form': form, 'show_loader': True})
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):

    if request.method == 'POST':
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            user.last_login_ip = request.META.get('REMOTE_ADDR')
            user.save()
            UserActivity.objects.create(
                user=user,
                activity_type='LOGIN',
                description='User logged in',
                ip_address=request.META.get('REMOTE_ADDR')
            )
            return redirect('home')
    else:
        form = UserLoginForm()
    return render(request, 'accounts/login.html', {'form': form, 'show_loader': True})
    return render(request, 'accounts/login.html', {'form': form})


@login_required
def logout_view(request):
    UserActivity.objects.create(
        user=request.user,
        activity_type='LOGOUT',
        description='User logged out',
        ip_address=request.META.get('REMOTE_ADDR')
    )
    logout(request)
    return redirect('accounts:login')

@method_decorator(staff_member_required, name='dispatch')
class UserActivityLogView(UserPassesTestMixin, ListView):
    model = UserActivity
    template_name = 'accounts/activity_log.html'
    context_object_name = 'activities'
    paginate_by = 50
    
    def test_func(self):
        return self.request.user.is_superuser

@staff_member_required
def user_management(request):
    users = User.objects.all()
    return render(request, 'accounts/user_management.html', {'users': users})

class UserManagementView(UserPassesTestMixin, ListView):
    model = User
    template_name = 'accounts/user_management.html'
    context_object_name = 'users'
    paginate_by = 50
    
    def test_func(self):
        return self.request.user.is_superuser

class EditUserView(UserPassesTestMixin, UpdateView):
    model = User
    form_class = UserEditForm
    template_name = 'accounts/edit_user.html'
    success_url = reverse_lazy('accounts:user_management')
    
    def test_func(self):
        return self.request.user.is_superuser
    
    def form_valid(self, form):
        messages.success(self.request, f"User {form.instance.username} has been updated successfully.")
        return super().form_valid(form)

@login_required
def delete_user(request, pk):
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to delete users.")
        return redirect('accounts:user_management')
    
    user = get_object_or_404(User, pk=pk)
    if user.is_superuser:
        messages.error(request, "Superuser accounts cannot be deleted.")
        return redirect('accounts:user_management')
    
    if request.method == 'POST':
        username = user.username
        user.delete()
        messages.success(request, f"User {username} has been deleted successfully.")
    
    return redirect('accounts:user_management')

