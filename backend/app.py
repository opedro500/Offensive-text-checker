from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import re
import os

# ==========================================
# 1. CONFIGURAÇÃO DA API
# ==========================================
app = FastAPI(
    title="Checador de Toxicidade",
    description="API para detectar linguagem ofensiva usando BERTimbau"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CARREGAMENTO DO MODELO (Em Memória)
# ==========================================
MODEL_PATH = "backend/model/bert_ofensivo"

print('Iniciando servidor e carregando o "cérebro" do BERT... Isso pode levar alguns segundos.')
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

# ==========================================
# 3. FUNÇÕES AUXILIARES (Nova Limpeza Leve)
# ==========================================
def limpar_texto_bert(texto):
    texto = str(texto).lower()
    # Remove APENAS Links e menções de usuário, mantendo a gramática para o BERT!
    texto = re.sub(r'http\S+|www\.\S+', '', texto)  
    texto = re.sub(r'@\w+', '', texto)             
    texto = re.sub(r'\brt\b', '', texto)           
    return texto.strip()

# Molde de como a requisição deve chegar
class TextoRequest(BaseModel):
    text: str

# ==========================================
# 4. ROTA PRINCIPAL DE PREDIÇÃO
# ==========================================
@app.post("/check")
async def checar_toxicidade(request: TextoRequest):
    try:
        # 1. Limpa o texto
        texto_limpo = limpar_texto_bert(request.text)
        
        # 2. Transforma o texto em números (Tensores)
        tokens = tokenizer(
            [texto_limpo],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="tf"
        )
        
        # 3. Pede para a IA fazer a previsão (Forma 100% segura extraindo como dict)
        saida = model({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens["token_type_ids"]
        })
        
        logits = saida.logits
        
        # 4. Transforma os números brutos (logits) em porcentagens
        probabilidades = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # Classe 0 = Seguro, Classe 1 = Ofensivo
        classe_vencedora = int(tf.argmax(logits, axis=1).numpy()[0])
        is_offensive = bool(classe_vencedora == 1)
        
        # Pega a % de certeza da classe que ganhou
        confianca = float(probabilidades[classe_vencedora])
        
        return {
            "original_text": request.text,
            "cleaned_text": texto_limpo,
            "is_offensive": is_offensive,
            "confidence": f"{confianca * 100:.2f}%"
        }
        
    except Exception as e:
        # Se algo der errado, a API não quebra! Ela devolve o erro para nós.
        print(f"ERRO INTERNO NA PREVISÃO: {e}")
        return {"error": str(e)}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")