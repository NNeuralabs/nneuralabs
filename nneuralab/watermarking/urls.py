
from django.urls import path
from .views import index, verify, example_marked, example_unmarked, example_model_extraction, register, logout_request, login_request
urlpatterns = [
    path('', index),
    path('verify/', verify.as_view(), name='verify'),
    path('verify/example-marked', example_marked.as_view(), name='example-marked'),
    path('verify/example-unmarked', example_unmarked.as_view(), name='example-unmarked'),
    path('example-extraction/', example_model_extraction.as_view(), name='example-extraction'),
    path('register/', register.as_view(), name='register'),
    path('logout/', logout_request.as_view(), name='logout'),
    path('login/', login_request.as_view(), name='login'),
]
