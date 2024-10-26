import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Dropout,
    BatchNormalization,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2


# Параметры
sequence_length = 10

# Загрузка данных
client_data = pd.read_csv(
    "train_data/cntrbtrs_clnts_ops_trn.csv",
    encoding='cp1251',
    delimiter=';',
    low_memory=False,
)
transaction_data = pd.read_csv(
    "train_data/trnsctns_ops_trn.csv",
    encoding='cp1251',
    delimiter=';',
    low_memory=False,
)

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

# Масштабирование числовых признаков
scaler = MinMaxScaler()
client_features = scaler.fit_transform(
    client_data[['prsnt_age', 'pnsn_age', 'cprtn_prd_d']]
)

# Предобработка данных транзакций
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

# Объединение данных клиентов и транзакций
merged_data = transaction_data.merge(client_data, on='accnt_id', how='left')

X_client = np.array(client_features)
y = np.array(client_data['erly_pnsn_flg'])

# Обработка последовательности транзакций
transaction_data = transaction_data.sort_values(['accnt_id', 'oprtn_date'])
transaction_seq_df = transaction_data.groupby('accnt_id').tail(sequence_length)
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

X_transactions = np.array(transaction_sequences)

# Разделение данных на обучающую и тестовую выборки
X_client_train, X_client_test, X_trans_train, X_trans_test, y_train, y_test = (
    train_test_split(
        X_client, X_transactions, y, test_size=0.2, random_state=42
    )
)

# Создание модели
client_input = Input(shape=(X_client.shape[1],))
transaction_input = Input(shape=(10, X_transactions.shape[2]))
transaction_lstm = LSTM(
    128, return_sequences=True, kernel_regularizer=l2(0.01)
)(transaction_input)
transaction_lstm = LSTM(64, kernel_regularizer=l2(0.01))(transaction_lstm)
transaction_lstm = BatchNormalization()(transaction_lstm)
transaction_lstm = Dropout(0.3)(transaction_lstm)
dense_client = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(
    client_input
)
dense_client = BatchNormalization()(dense_client)
dense_client = Dropout(0.3)(dense_client)
concat = concatenate([dense_client, transaction_lstm])
output = Dense(1, activation='sigmoid')(concat)

model = Model(inputs=[client_input, transaction_input], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
with tf.device('/GPU:0'):
    model.fit(
        [X_client_train, X_trans_train],
        y_train,
        validation_split=0.2,
        epochs=250,
        batch_size=32,
        callbacks=[early_stopping, checkpoint],
    )


loss, accuracy = model.evaluate([X_client_test, X_trans_test], y_test)
print(f'Accuracy: {accuracy:.4f}')
