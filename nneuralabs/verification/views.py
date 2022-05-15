from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

class Prediction(APIView):
    def get(self, request):
        return render(request, 'verification/index.html') 
    def post(self, request):
        data = request.data
        return Response(data, status=200)
