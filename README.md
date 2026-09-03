Projeto: Classificador de Comentários Ofensivos (Substituto da Perspective API)

Este projeto é uma versão renovada de um antigo sistema de moderação de texto. A iniciativa de reconstruí-lo do zero surgiu após a descontinuação da Perspective API do Google, o que nos motivou a criar uma solução de inteligência artificial totalmente independente, de código aberto, executada localmente e especializada na língua portuguesa.

**A Arquitetura de 3 Níveis**
O sistema permite classificar a toxicidade de um texto alternando em tempo real entre três abordagens de Machine Learning:

* Nível 1 (Clássico): TF-IDF + Regressão Logística. Uma abordagem estatística leve e ultrarrápida baseada na frequência de palavras, com limpeza agressiva de stopwords.
* Nível 2 (Sequencial): Rede Neural Recorrente (GRU). Lê o texto da esquerda para a direita retendo memória de curto prazo, permitindo a compreensão de gírias e do contexto básico da frase.
* Nível 3 (Estado da Arte): BERTimbau (Transformer). Um modelo colossal de 110 milhões de parâmetros treinado pela NeuralMind. O modelo passou por um Fine-Tuning específico para entender o sarcasmo e a semântica profunda da ofensa no vocabulário brasileiro.

**Como Executar o Projeto (Docker)**
A aplicação foi construída para rodar em um container unificado que já atende a API e a interface gráfica simultaneamente.

1. Construa a imagem Docker na raiz do projeto (isso instalará as dependências e o pacote NLTK automaticamente):
docker build -t checador-ofensas .

2. Inicie o container liberando a porta local:
docker run -p 8000:8000 checador-ofensas

3. Acesse o sistema abrindo o seu navegador no endereço:
http://localhost:8000

**Stack Tecnológica**
* Backend: Python 3.10, FastAPI, Uvicorn
* Machine Learning: TensorFlow/Keras, Hugging Face Transformers, Scikit-Learn, NLTK
* Frontend: HTML, CSS Flexbox, Vanilla JavaScript, FontAwesome
* MLOps (Treinamento prévio): Weights & Biases (WandB Sweep e Bayes Optimization)