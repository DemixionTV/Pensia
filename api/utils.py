import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import io

# Параметры
sequence_length = 10
output_file = 'predictions.csv'


# Загрузка и предобработка новых данных
def preprocess_data(file_client, file_trans):
    client_data = pd.read_csv(
        file_client, encoding='cp1251', delimiter=';', low_memory=False
    )
    transaction_data = pd.read_csv(
        file_trans, encoding='cp1251', delimiter=';', low_memory=False
    )

    label_encoders = {}
    for col in [
        'gndr',
        'accnt_status',
        'addrss_type',
        'rgn',
        'dstrct',
        'city',
        'sttlmnt',
        'okato',
        'prvs_npf',
    ]:
        le = LabelEncoder()
        client_data[col] = le.fit_transform(client_data[col])
        label_encoders[col] = le
    scaler = MinMaxScaler()
    client_features = scaler.fit_transform(
        client_data[['prsnt_age', 'pnsn_age', 'cprtn_prd_d']]
    )

    transaction_data['mvmnt_type'] = LabelEncoder().fit_transform(
        transaction_data['mvmnt_type']
    )
    transaction_data['sum_type'] = LabelEncoder().fit_transform(
        transaction_data['sum_type']
    )
    transaction_data['sum'] = scaler.fit_transform(transaction_data[['sum']])
    transaction_data['oprtn_month'] = pd.to_datetime(
        transaction_data['oprtn_date']
    ).dt.month
    transaction_data['oprtn_year'] = pd.to_datetime(
        transaction_data['oprtn_date']
    ).dt.year

    transaction_data = transaction_data.sort_values(['accnt_id', 'oprtn_date'])
    transaction_seq_df = transaction_data.groupby('accnt_id').tail(
        sequence_length
    )
    transaction_sequences = (
        transaction_seq_df.groupby('accnt_id')
        .apply(
            lambda x: x[
                ['mvmnt_type', 'sum_type', 'sum', 'oprtn_month', 'oprtn_year']
            ].values[-sequence_length:]
        )
        .reindex(client_data['accnt_id'])
        .apply(
            lambda x: np.pad(
                x, ((sequence_length - len(x), 0), (0, 0)), 'constant'
            )
        )
        .tolist()
    )
    X_client = np.array(client_features)
    X_transactions = np.array(transaction_sequences)

    return X_client, X_transactions, client_data


model = load_model('best_model.h5')


def make_predictions(input_client_file, input_trans_file, output_file):
    X_client, X_transactions, client_data = preprocess_data(
        input_client_file, input_trans_file
    )
    predictions = model.predict([X_client, X_transactions])
    print(predictions.flatten())
    output_df = pd.DataFrame(
        {
            'accnt_id': client_data['accnt_id'],
            'erly_pnsn_flg': [
                1 if pred > 0.5 else 0 for pred in predictions.flatten()
            ],
        }
    )
    output_df.to_csv(output_file, index=False,sep=',', encoding='utf-8')
