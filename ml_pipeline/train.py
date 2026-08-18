import os
import re
import pandas as pd
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from keras.optimizers import Adam
import wandb
from wandb.integration.keras import WandbMetricsLogger

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS
# ==========================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5 
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Inicializando o WandB
wandb.init(
    project="checador-texto-ofensivo",
    name="bertimbau-fine-tuning",
    config={
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "max_len": MAX_LEN
    }
)

# ==========================================
# 2. FUNÇÃO DE LIMPEZA
# ==========================================
nltk.download('stopwords', quiet=True)
STOP_WORDS_PT = set(stopwords.words('portuguese'))

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\.\S+', '', texto)  
    texto = re.sub(r'@\w+', '', texto)             
    texto = re.sub(r'\brt\b', '', texto)           
    texto = re.sub(r'\d+', '', texto)              
    texto = re.sub(r'[^\w\s]', '', texto)          
    texto = texto.replace('_', '')                 
    
    palavras = texto.split()
    palavras_limpas = [p for p in palavras if p not in STOP_WORDS_PT]
    return ' '.join(palavras_limpas)

# ==========================================
# 3. CARREGAMENTO DOS DADOS (ToLD-BR)
# ==========================================
print("Carregando o dataset ToLD-BR...")
df = pd.read_csv("ml_pipeline/data/ToLD-BR.csv")

print("Gerando labels e limpando textos...")
colunas_de_toxicidade = ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']
df['label'] = (df[colunas_de_toxicidade].max(axis=1) > 0).astype(int)

df['text'] = df['text'].apply(limpar_texto)

textos = df['text'].tolist()
labels = df['label'].tolist()

# ==========================================
# 4. TOKENIZAÇÃO DO BERT
# ==========================================
print(f"Baixando e aplicando o tokenizador do {MODEL_NAME}...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

tokens = tokenizer(
    textos,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="tf"
)

# ==========================================
# 5. DIVISÃO EM TREINO E VALIDAÇÃO
# ==========================================
print("Montando os Tensores...")
dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": tokens["input_ids"],
        "token_type_ids": tokens["token_type_ids"],
        "attention_mask": tokens["attention_mask"]
    },
    labels
))

tamanho_total = len(textos)
tamanho_treino = int(0.8 * tamanho_total)

dataset = dataset.shuffle(buffer_size=10000, seed=SEED)
train_dataset = dataset.take(tamanho_treino).batch(BATCH_SIZE)
val_dataset = dataset.skip(tamanho_treino).batch(BATCH_SIZE)

# ==========================================
# 6. CARREGANDO O MODELO BERT
# ==========================================
print("Baixando a arquitetura pré-treinada do BERTimbau...")
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=['accuracy']
)

# ==========================================
# 7. INICIANDO O TREINAMENTO
# ==========================================
# Configurando o EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=2, # Para se não melhorar por 2 épocas seguidas
    restore_best_weights=True
)

print("Iniciando o Fine-Tuning...")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[WandbMetricsLogger(), early_stop],
    verbose=1
)

# Finaliza a sessão do WandB
wandb.finish()

# ==========================================
# 8. SALVANDO O MODELO PARA O BACKEND
# ==========================================
pasta_destino = "backend/model/bert_ofensivo"
os.makedirs(pasta_destino, exist_ok=True)

print(f"\nSalvando o modelo e o tokenizador na pasta '{pasta_destino}'...")
model.save_pretrained(pasta_destino)
tokenizer.save_pretrained(pasta_destino)

print("Tudo pronto! Seu modelo BERT está treinado e salvo.")