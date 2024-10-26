from drf_yasg import openapi

CSV_FILE_RESPONSE = openapi.Response(
    'CSV file',
    schema=openapi.Schema(type=openapi.TYPE_FILE),
)

VALIDATION_ERROR_RESPONSE = openapi.Response(
    description='Ошибка валидации входных данных',
    schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'field_name': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Неправильный формат данных',
                example=['Файл должен быть в формате CSV.']
            )
        }
    )
)
