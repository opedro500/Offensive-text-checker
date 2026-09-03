import pandas as pd
import wandb
import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from wandb.integration.keras import WandbCallback
from sklearn.model_selection import train_test_split

wandb.init(project="Offensive-text-checker-ml_pipeline")
config = wandb.config

df_train = pd.read_csv("ml_pipeline/data/train_data.csv")
df_test = pd.read_csv("ml_pipeline/data/test_data.csv")

X = df_train['text'].fillna('')
y = df_train['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# learning_rate = config.get('learning_rate', 2e-5)
# batch_size = config.get('batch_size', 16)
# epochs = 4
# max_length = 64
# model_name = "neuralmind/bert-base-portuguese-cased"

learning_rate = 3e-5
batch_size = 16
epochs = 4
max_length = 64
model_name = "neuralmind/bert-base-portuguese-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(X_train.tolist(), truncation=True, padding='max_length', max_length=max_length)
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding='max_length', max_length=max_length)
test_encodings = tokenizer(df_test['text'].fillna('').tolist(), truncation=True, padding='max_length', max_length=max_length)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), df_test['label'].values)).batch(batch_size)

model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, from_pt=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.0)]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[WandbCallback(save_model=False)]
)

loss_val, test_accuracy = model.evaluate(test_dataset)
wandb.log({"test_accuracy": test_accuracy})

os.makedirs("backend/models/level3_bert", exist_ok=True)
model.save_pretrained("backend/models/level3_bert")
tokenizer.save_pretrained("backend/models/level3_bert")

wandb.finish()