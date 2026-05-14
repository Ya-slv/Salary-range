
from django.urls import path
from . import views

urlpatterns = [
    path('parse_resume/', views.parse_resume),
    path('calculate_salary/', views.calculate_salary)
]
