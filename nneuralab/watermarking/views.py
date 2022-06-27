from turtle import pd
from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import pickle
from rest_framework.views import APIView
from rest_framework.response import Response
import torch
from torch import nn, threshold
from torch.nn import functional as F
import requests
import json
import numpy as np
import random as rand
from scipy.special import comb
from math import floor, sqrt
from watermarking.utils import LeNet, Net
from torch.utils.data import DataLoader
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
import gzip
import os

#from scipy.special import comb
# Create your views here.


def index(request):
    return render(request, 'landing.html', {})

class login_request(View):
    def get(self, request):
        return render(request, 'login.html', {})
    
    def post(self, request):
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return render(request, 'landing.html', {})

        return render(request, 'login.html', {})


class register(View):
    def get(self, request):
        return render(request, 'register.html', {})

    def post(self, request):
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return render(request, 'register_success.html', {})
        return self.get(request)


class logout_request(View):
    def get(self, request):
        logout(request)
        return render(request, 'landing.html', {})

   
@method_decorator(csrf_exempt, name='dispatch')
class verify(View):
    def get(self, request):
        return render(request, 'verification.html', {})

    def post(self, request):
        batch_size = 1
        data = request.POST
        ownership = request.FILES["keys_values"]
        fs = FileSystemStorage()
        filename = fs.save(ownership.name, ownership)
        with open(os.path.join('public/media/', filename), 'rb') as f:
            data = pickle.load(f)
        
        predictions_reference = data['labels']
        keys = data['inputs']

        url = request.POST['api_link']
        key_loader = DataLoader(keys, batch_size=batch_size, shuffle=False)
        pred_suspect = []
        for _ , batch in enumerate(key_loader):
            data_list = batch.tolist()
            response = requests.post(url, json=data_list)
            
            pred_suspect += list(response.json())

        
        trigger_size = 100
        number_labels = 10
        error_rate = 0.001
        precision = 1 / trigger_size
        threshold = 1 / number_labels
        S = 0

        for i in range(0, int(threshold * trigger_size) + 1):
            wrong_detected = ((1 - 1 / number_labels)**(trigger_size - i))
            S += comb(trigger_size, i) * (1 / (number_labels**i)) * wrong_detected
        while S < (1 - error_rate) or threshold == 1:
            old_threshold = threshold
            threshold += precision
            old_bound = floor(old_threshold * trigger_size) + 1
            new_bound = floor(threshold * trigger_size) + 1
            for i in range(old_bound, new_bound):
                wrong_detected = (1 - 1 / number_labels)**(trigger_size - i)
                S += comb(trigger_size, i) * \
                (1 / number_labels**i) * wrong_detected


        threshold = min(threshold, 1)
        counter = 0
        for label, value in zip(pred_suspect,predictions_reference):
            if label == value:
                counter += 1
        watermarked = False
        if counter/len(pred_suspect) >= threshold:
            watermarked = True
        print(counter/len(pred_suspect))
        return render(request, 'verification_result.html', {'watermarked': watermarked})


class example_marked_custom(APIView):
    def post(self, request):
        data = request.data
        suspect_model = LeNet()
        weights = torch.load('models/marked_custom_model.pt')
        suspect_model.load_state_dict(weights)
        with torch.no_grad():
            suspect_model.eval()
            data = torch.Tensor(data)
            result = suspect_model(data)
            pred = []
            for res in result:
                pred.append(int(torch.argmax(res)))  
        return Response(pred, status=200)


class example_unmarked_custom(APIView):
    def post(self, request):
        data = request.data
        suspect_model = LeNet()
        weights = torch.load('models/clean_custom_model.pt')
        suspect_model.load_state_dict(weights)
        with torch.no_grad():
            suspect_model.eval()
            data = torch.Tensor(data)
            result = suspect_model(data)
            pred = []
            for res in result:
                pred.append(int(torch.argmax(res)))  
        return Response(pred, status=200)

class example_marked_noise(APIView):
    def post(self, request):
        data = request.data
        suspect_model = Net()
        weights = torch.load('models/marked_noise_model.pt')
        suspect_model.load_state_dict(weights)
        with torch.no_grad():
            suspect_model.eval()
            data = torch.Tensor(data)
            result = suspect_model(data)
            pred = []
            for res in result:
                pred.append(int(torch.argmax(res)))       
        return Response(pred, status=200)

class example_unmarked_noise(APIView):
    def post(self, request):
        data = request.data
        suspect_model = Net()
        weights = torch.load('models/model.pt')
        suspect_model.load_state_dict(weights)
        with torch.no_grad():
            suspect_model.eval()
            data = torch.Tensor(data)
            result = suspect_model(data)
            pred = []
            for res in result:
                pred.append(int(torch.argmax(res)))
        return Response(pred, status=200)

class demo_tutorial(View):
    def get(self, request):
        return render(request, 'demo_tutorial.html', {})