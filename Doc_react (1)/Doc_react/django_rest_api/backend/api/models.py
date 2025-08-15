from django.db import models

# Create your models here.
class UploadFiles(models.Model):
    uploaded_at = models.DateTimeField(null=True, blank = True)
    job_description = models.TextField(blank=True, null=True)
    uploadedfile=models.FileField(upload_to='file_upload/',null=True, blank=True)