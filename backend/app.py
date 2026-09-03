from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# O download do NLTK já foi feito no Dockerfile, mas garantimos aqui
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('portuguese'))

app = FastAPI(title="API de Classificação de Ofensas")

# --- Carregamento dos Modelos ---
vec_l1 = joblib.load("backend/models/level1_tfidf/vectorizer.pkl")
mod_l1 = joblib.load("backend/models/level1_tfidf/model.pkl")

with open("backend/models/level2_gru/tokenizer.pkl", "rb") as f:
    tok_l2 = pickle.load(f)
mod_l2 = tf.keras.models.load_model("backend/models/level2_gru/model.h5")

tok_l3 = AutoTokenizer.from_pretrained("backend/models/level3_bert")
mod_l3 = TFAutoModelForSequenceClassification.from_pretrained("backend/models/level3_bert")

# --- Estrutura dos Dados (Pydantic) ---
class PredictRequest(BaseModel):
    text: str
    level: str = "3"

# --- Funções de Limpeza ---
def limpar_texto_base(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'\brt\b', '', texto)
    return texto.strip()

def limpar_texto_agressivo(texto):
    texto = limpar_texto_base(texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    return ' '.join([p for p in texto.split() if p not in stop_words])

# --- Rotas da API ---
@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio")

    if req.level == '1':
        texto_limpo = limpar_texto_agressivo(req.text)
        X = vec_l1.transform([texto_limpo])
        score = float(mod_l1.predict_proba(X)[0][1])
        
    elif req.level == '2':
        texto_limpo = limpar_texto_base(req.text)
        seq = tok_l2.texts_to_sequences([texto_limpo])
        pad = pad_sequences(seq, maxlen=64, padding='pre', truncating='post')
        score = float(mod_l2.predict(pad, verbose=0)[0][0])
        
    elif req.level == '3':
        texto_limpo = limpar_texto_base(req.text)
        encodings = tok_l3([texto_limpo], truncation=True, padding='max_length', max_length=64, return_tensors='tf')
        logits = mod_l3(encodings).logits
        score = float(tf.nn.sigmoid(logits)[0][0])
        
    else:
        raise HTTPException(status_code=400, detail="Nível inválido")

    is_offensive = bool(score > 0.5)
    
    return {
        "offensive": is_offensive,
        "confidence": round(score * 100, 2),
        "level_used": req.level
    }

# --- Servindo os arquivos estáticos do Frontend ---
app.mount("/styles", StaticFiles(directory="frontend/styles"), name="styles")
app.mount("/javascript", StaticFiles(directory="frontend/javascript"), name="javascript")

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")