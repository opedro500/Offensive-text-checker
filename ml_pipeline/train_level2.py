import pandas as pd
import wandb
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbCallback
from sklearn.model_selection import train_test_split

wandb.init(project="Offensive-text-checker-ml_pipeline")
config = wandb.config

df_train = pd.read_csv("ml_pipeline/data/train_data.csv")
df_test = pd.read_csv("ml_pipeline/data/test_data.csv")

X = df_train['text'].fillna('')
y = df_train['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# vocab_size = config.get('vocab_size', 5000)
# max_length = 64
# embedding_dim = config.get('embedding_dim', 64)
# gru_units = config.get('gru_units', 64)
# learning_rate = config.get('learning_rate', 0.001)
# batch_size = config.get('batch_size', 32)
# dropout_rate = config.get('dropout_rate', 0.3)
# epochs = 5

# Valores achados pelo sweep no wandb
vocab_size = 5000
max_length = 64
embedding_dim = 64
gru_units = 32
learning_rate = 0.001
batch_size = 32
dropout_rate = 0.5
epochs = 5

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='pre', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding='pre', truncating='post')

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    SpatialDropout1D(dropout_rate),
    GRU(gru_units, dropout=dropout_rate),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[WandbCallback(save_model=False)]
)

X_test = df_test['text'].fillna('')
y_test = df_test['label'].values
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='pre', truncating='post')

loss, test_accuracy = model.evaluate(X_test_pad, y_test)
wandb.log({"test_accuracy": test_accuracy})

os.makedirs("backend/models/level2_gru", exist_ok=True)
model.save("backend/models/level2_gru/model.h5")
with open("backend/models/level2_gru/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

wandb.finish()