from rest_framework import serializers


class CSVFilesSerializer(serializers.Serializer):
    clients = serializers.FileField(help_text='CSV файл с клиентами')
    transactions = serializers.FileField(help_text='CSV файл с транзакциями')

    def validate_clients(self, value):
        return self.validate_csv_file(value)

    def validate_transactions(self, value):
        return self.validate_csv_file(value)

    def validate_csv_file(self, file):
        if not file.name.endswith('.csv'):
            raise serializers.ValidationError('Файл должен быть в формате CSV.')
        return file
