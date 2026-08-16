import pandas as pd
import tensorflow as tf
from keras import layers
import keras
import wandb
from wandb.integration.keras import WandbMetricsLogger

# ==========================================
# 1. PREPARAÇÃO DOS DADOS (ToLD-BR)
# ==========================================
print("Carregando o dataset ToLD-BR...")
df = pd.read_csv("ml_pipeline/data/ToLD-BR.csv") # Ajuste o caminho se necessário

# Criando a label binária (1 = Tóxico, 0 = Seguro)
colunas_de_toxicidade = ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']
df['label'] = (df[colunas_de_toxicidade].max(axis=1) > 0).astype(int)

# Pegando textos e labels
textos = df['text'].tolist()
labels = df['label'].tolist()

# Convertendo para tf.data.Dataset para alta performance na GPU
dataset = tf.data.Dataset.from_tensor_slices((textos, labels))

# Embaralhando e dividindo: 80% Treino, 20% Validação
tamanho_total = len(textos)
tamanho_treino = int(0.8 * tamanho_total)

dataset = dataset.shuffle(buffer_size=10000)
train_ds = dataset.take(tamanho_treino).batch(32)
val_ds = dataset.skip(tamanho_treino).batch(32)

# ==========================================
# 2. VETORIZAÇÃO (Inspirado no seu professor)
# ==========================================
vocab_size = 10000
max_len = 64

vetorizador = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_len
)

print("Adaptando o vocabulário (isso pode levar alguns segundos)...")
# O vetorizador aprende as palavras usando apenas os textos (sem as labels)
vetorizador.adapt(train_ds.map(lambda text, label: text))

# Aplicando a vetorização aos datasets
def vetorizar_texto(texto, label):
    return vetorizador(texto), label

train_ds = train_ds.map(vetorizar_texto, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(vetorizar_texto, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. CLASSES DO TRANSFORMER
# ==========================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, input_dim, output_dim):
        super().__init__()
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=max_len, output_dim=output_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, padding_mask=None):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=padding_mask)
        x = self.norm1(x + attn_output)
        return self.norm2(x + self.ffn(x))

def build_classifier(vocab_size, max_len, embed_dim, dense_dim, num_heads, n_layers):
    inputs = keras.Input(shape=(max_len,), dtype="int64")
    x = PositionalEmbedding(max_len, vocab_size, embed_dim)(inputs)
    
    for _ in range(n_layers):
        x = EncoderLayer(embed_dim, dense_dim, num_heads)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

# ==========================================
# 4. CONFIGURAÇÃO WANDB E TREINAMENTO
# ==========================================
config = {
    "vocab_size": vocab_size,
    "max_len": max_len,
    "embed_dim": 64,     # Experimente mudar para 128
    "dense_dim": 128,    # Experimente mudar para 256
    "num_heads": 4,      # Experimente mudar para 8
    "n_layers": 2,       # Experimente mudar para 3 ou 4
    "epochs": 10
}

wandb.init(project="checador-texto-ofensivo", config=config, name="transformer-base")

model = build_classifier(
    config["vocab_size"], config["max_len"], config["embed_dim"], 
    config["dense_dim"], config["num_heads"], config["n_layers"]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Iniciando o treinamento na RTX 3060...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["epochs"],
    callbacks=[WandbMetricsLogger()]
)

# Salva o modelo pronto
model.save("../backend/model/meu_transformer.keras")
wandb.finish()