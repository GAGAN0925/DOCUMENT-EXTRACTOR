from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .src.main import convert_file_to_images, process_images_with_layoutparser
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.middleware.csrf import get_token


from .serializers import UploadFileSerializer


class UploadFileView(APIView):
    serializer_class = UploadFileSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        csrf_token = get_token(request)
        return JsonResponse({'csrf_token':csrf_token})
    
    def post(self, request):
        serializer =self.serializer_class(data=request.data)  

        if serializer.is_valid():
            serializer.save()   
            file_path = serializer.data['uploadedfile'] 
            file_path = file_path[1:]  

            filename = os.path.basename(file_path)
            #output_folder = 'project-1/project-1/public/media/extracted_data/'+filename+'/'
            output_folder = 'media/extracted_data/'+filename+'/'
            os.makedirs(output_folder, exist_ok=True)
            
            images_array = convert_file_to_images(file_path, output_folder)
            resultdict = process_images_with_layoutparser(images_array, output_folder)           
            

            #return Response(serializer.data)
            return Response(resultdict)
            #return JsonResponse({'message':'This is getting really difficult'})

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        #return Response(file, status=status.HTTP_400_BAD_REQUEST)