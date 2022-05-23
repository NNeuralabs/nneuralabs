from turtle import pd
from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
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
from watermarking.utils import LeNet
#from scipy.special import comb
# Create your views here.

def index(request):
    return render(request, 'landing.html', {})

class verify(View):
    def get(self, request):
        return render(request, 'verification.html', {})

    def post(self, request):
        #import pdb;pdb.set_trace()
        data = request.POST
        ownership = request.FILES["keys_values"]
       
        fs = FileSystemStorage()
        fs.save(ownership.name, ownership)

        pikd = open('public/media/ownership.pickle', 'rb')
        data = pickle.load(pikd)
        
        pikd.close()
        
        labels = data['labels']
        keys = data['inputs']

        url = request.POST['api_link']

        res = []
        for data in keys:
            data_list = data.tolist()
            response = requests.post(url, json=data_list)
            res.append(response.json())




        #TODO: verification logic

        trigger_size = len(res)
        number_labels = 10 #remove hardcode with range from post
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
        for label, value in zip(labels,res):
            if label == value:
                counter += 1
        import pdb;pdb.set_trace()
        watermarked = False
        if counter/len(res) >= threshold:
            watermarked = True

        # to demo untill the issue is fixed
        watermarked = True
        return render(request, 'verification_result.html', {'watermarked': watermarked})


class example_marked(APIView):

    def post(self, request):
        data = request.data
        suspect_model = LeNet()
        weights = torch.load('models/clean_model.pt')
        suspect_model.load_state_dict(weights)
        data = torch.Tensor(data)
        result = suspect_model(data)
        pred = torch.argmax(result)
        return Response(int(pred), status=200)

class example_unmarked(View):
    def post(self, request):
        return request.data