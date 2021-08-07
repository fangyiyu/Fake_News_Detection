from django.urls import include, path
from . import views

app_name = 'detector'

urlpatterns = [
    path("", views.index, name="index"),
    path('pred/', views.pred, name="pred"),
]