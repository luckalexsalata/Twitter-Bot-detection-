from django.contrib import admin
from django.urls import path
#from Bot_detection import views
from . import views
urlpatterns = [
   #path('', views.bot, name = 'bot'),
   path('get_res/', views.get_res, name='get_res'),
]
