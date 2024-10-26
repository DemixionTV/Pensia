from rest_framework.viewsets import GenericViewSet
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from drf_yasg.utils import swagger_auto_schema

from .utils import make_predictions
from .serializers import CSVFilesSerializer
from .responses import CSV_FILE_RESPONSE, VALIDATION_ERROR_RESPONSE


class CSVProcessViewSet(GenericViewSet):
    """
    Обработка файлов

    Обработка файлов
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

        result_csv = make_predictions(clients, transactions, 'result.csv')

        response = Response(result_csv, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="result.csv"'

        return response
