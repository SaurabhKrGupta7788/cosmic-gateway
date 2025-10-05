from django.contrib import admin
from django.urls import path, include
from .views import *
from . import views

urlpatterns = [
    path('', main, name='main'),
    path('koi/', k2, name='k2'),
    path('predict/', predict_koi, name='predict_koi'),
    path('train/', views.tune_model_view, name='tune_model'),
    path('about/', about, name='about'),
    path('how_it_works/', how_it_works, name='how_it_works'),
    path('predict_train/', views.predict_koi_train, name='predict_koi_train'),
]
