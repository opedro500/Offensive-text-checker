import pandas as pd
import wandb
import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('portuguese'))

def limpar_texto_agressivo(texto):
    texto = str(texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = texto.split()
    return ' '.join([p for p in palavras if p not in stop_words])

wandb.init(project="Offensive-text-checker-ml_pipeline")
config = wandb.config

df_train = pd.read_csv("ml_pipeline/data/train_data.csv")
df_test = pd.read_csv("ml_pipeline/data/test_data.csv")

df_train['text'] = df_train['text'].apply(limpar_texto_agressivo)
df_test['text'] = df_test['text'].apply(limpar_texto_agressivo)

X = df_train['text'].fillna('')
y = df_train['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# vectorizer = TfidfVectorizer(
#     max_features=config.get('max_features', 5000), 
#     ngram_range=(1, config.get('ngram_max', 1))
# )

# Valores achados pelo sweep no wandb
vectorizer = TfidfVectorizer(
    max_features=5000, 
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# model = LogisticRegression(
#     C=config.get('C', 1.0), 
#     solver=config.get('solver', 'lbfgs'), 
#     max_iter=1000
# )

# Valores achados pelo sweep no wandb
model = LogisticRegression(
    C=10.0, 
    solver='lbfgs', 
    max_iter=1000
)

model.fit(X_train_vec, y_train)
val_preds = model.predict(X_val_vec)
val_accuracy = accuracy_score(y_val, val_preds)

wandb.log({"val_accuracy": val_accuracy})

X_full_vec = vectorizer.fit_transform(X)
model.fit(X_full_vec, y)

X_test = df_test['text'].fillna('')
y_test = df_test['label']
X_test_vec = vectorizer.transform(X_test)

test_preds = model.predict(X_test_vec)
test_accuracy = accuracy_score(y_test, test_preds)

wandb.log({"test_accuracy": test_accuracy})

os.makedirs("backend/models/level1_tfidf", exist_ok=True)
joblib.dump(vectorizer, "backend/models/level1_tfidf/vectorizer.pkl")
joblib.dump(model, "backend/models/level1_tfidf/model.pkl")

wandb.finish()