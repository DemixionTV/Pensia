import os
from rest_framework.viewsets import GenericViewSet
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from django.core.files.storage import default_storage
from drf_yasg.utils import swagger_auto_schema

from .utils import make_predictions
from .serializers import CSVFilesSerializer
from .responses import CSV_FILE_RESPONSE, VALIDATION_ERROR_RESPONSE


class CSVProcessViewSet(GenericViewSet):
    """
    Обработка файлов для прогнозирования

    Обработка файлов на основе данных машинного обучения
    """
    parser_classes = (MultiPartParser, FormParser, FileUploadParser)
    serializer_class = CSVFilesSerializer

    @swagger_auto_schema(
        request_body=CSVFilesSerializer,
        responses={200: CSV_FILE_RESPONSE,
                   400: VALIDATION_ERROR_RESPONSE}
    )
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        clients = serializer.validated_data['clients']
        transactions = serializer.validated_data['transactions']

        path_clients = os.path.join('uploads/', clients.name)
        full_path_clients = f'media/{default_storage.save(path_clients, clients)}'
        path_transactions = os.path.join('uploads/', transactions.name)
        full_path_transactions = f'media/{default_storage.save(path_transactions, transactions)}'

        make_predictions(full_path_clients, full_path_transactions, 'result.csv')
        with open('result.csv', 'r',encoding='utf-8') as f:
            csv_content = f.read()
        response = HttpResponse(csv_content, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="result.csv"'

        return response
