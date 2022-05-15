from django.urls import path
from .views import Prediction

urlpatterns = [
    path('', Prediction.as_view(), name = 'prediction'),
]