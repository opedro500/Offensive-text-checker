import pandas as pd
import re

# A nossa nova função focada no BERT
def limpar_texto_bert(texto):
    texto = str(texto).lower()
    # Remove APENAS Links e menções de usuário
    texto = re.sub(r'http\S+|www\.\S+', '', texto)  
    texto = re.sub(r'@\w+', '', texto)             
    texto = re.sub(r'\brt\b', '', texto)  
    return texto.strip()

print("Carregando o dataset HateBR...")
df = pd.read_csv("ml_pipeline/data/HateBR.csv")

# Pegando 5 exemplos aleatórios
exemplos = df.sample(5, random_state=42)

print("\n" + "="*60)
print("COMPARANDO A LIMPEZA (HateBR + BERT)")
print("="*60)

for i, (_, linha) in enumerate(exemplos.iterrows(), 1):
    # No HateBR a coluna de texto se chama 'comentario'
    texto_original = linha['comentario'] 
    texto_limpo = limpar_texto_bert(texto_original)
    
    print(f"\n--- EXEMPLO {i} ---")
    print(f"🔴 ANTES : {texto_original}")
    print(f"🟢 DEPOIS: {texto_limpo}")