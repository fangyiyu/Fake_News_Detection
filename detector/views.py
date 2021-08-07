from django.shortcuts import render, redirect
from . import predict
import pandas as pd


# Create your views here.

def index(request):
    return render(request, 'detector/index.html')

def pred(request):
    if request.method == "POST":
        original_text = request.POST.get('text')
        prediction = predict.predictor(original_text)
        return render(request, 'detector/index.html', {"result": prediction,
        "text": original_text,})





