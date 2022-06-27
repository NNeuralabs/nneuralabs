
from django.urls import path
from .views import index, verify, example_marked_custom, example_unmarked_custom,  example_marked_noise, example_unmarked_noise, register, logout_request, login_request, demo_tutorial
urlpatterns = [
    path('', index),
    path('verify/', verify.as_view(), name='verify'),
    path('verify/tutorial/', demo_tutorial.as_view(), name='tutorial'),
    path('verify/example-marked-custom', example_marked_custom.as_view(), name='example-marked-custom'),
    path('verify/example-unmarked-custom', example_unmarked_custom.as_view(), name='example-unmarked-custom'),
    path('verify/example-marked-noise', example_marked_noise.as_view(), name='example-marked-noise'),
    path('verify/example-unmarked-noise', example_unmarked_noise.as_view(), name='example-unmarked-noise'),
    path('register/', register.as_view(), name='register'),
    path('logout/', logout_request.as_view(), name='logout'),
    path('login/', login_request.as_view(), name='login'),
]
