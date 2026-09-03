import pandas as pd
import re
from sklearn.model_selection import train_test_split

def limpar_texto_base(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\.\S+', '', texto) # Remove links
    texto = re.sub(r'@\w+', '', texto)             # Remove @usuarios
    texto = re.sub(r'\brt\b', '', texto)           # Remove o 'RT' do Twitter
    return texto.strip()

df = pd.read_csv("ml_pipeline/data/HateBR.csv")

df = df[['comentario', 'label_final']].rename(columns={'comentario': 'text', 'label_final': 'label'})

df['text'] = df['text'].apply(limpar_texto_base)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv("ml_pipeline/data/train_data.csv", index=False)
test_df.to_csv("ml_pipeline/data/test_data.csv", index=False)