from rest_framework import serializers
from .models import UploadFiles

class UploadFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadFiles
        fields = ['uploadedfile']