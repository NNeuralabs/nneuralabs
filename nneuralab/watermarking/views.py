from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
import pickle
from rest_framework.views import APIView
from rest_framework.response import Response
import torch
from torch import nn
from torch.nn import functional as F
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


        res = []
        for data in keys:
            res.append(request.post(data["api_link"], params=None, headers=None, data=image_data))

        print(keys)

        print(labels)
        import pdb;pdb.set_trace()
        return render(request, 'landing.html', {})


class example_marked(APIView):

    def post(self, request):
        print(request.POST)
        model = torch.load('models/watermarked.zip') 

        prediction = model.predict(request.POST)
        response_dict = {'prediction': prediction }
        return Response(response_dict, status=200)


class example_unmarked(View):
    def post(self, request):
        return request.data