from django.urls import path
from . import views

urlpatterns = [    
    path('uploadfile/',views.UploadFileView.as_view())
]
