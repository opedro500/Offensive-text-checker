# Usa um Linux levinho com Python 3.10
FROM python:3.10-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OTIMIZAÇÃO: Baixa as stopwords no momento da build da imagem
RUN python -m nltk.downloader stopwords

# Copia a pasta do backend (com o modelo dentro) e do frontend
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Libera a porta 8000
EXPOSE 8000

# Comando para ligar o servidor quando o container iniciar
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]