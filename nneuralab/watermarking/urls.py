
from django.urls import path
from .views import index, verify, example_marked, example_unmarked
urlpatterns = [
    path('', index),
    path('verify/', verify.as_view(), name='verify'),
    path('verify/example-marked', example_marked.as_view(), name='example-marked'),
    path('verify/example-unmarked', example_unmarked.as_view(), name='example-unmarked'),
]
