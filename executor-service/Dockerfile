# Usa uma imagem base Python slim
FROM python:3.10-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de requisitos primeiro (aproveita cache do Docker)
COPY requirements.txt ./

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código do aplicativo
COPY main.py ./

# Expõe a porta que o Flask vai usar (o Cloud Run espera 8080 por padrão)
EXPOSE 8080

# Comando para rodar o aplicativo Flask quando o contêiner iniciar
# Usa 'python -u' para garantir que os prints apareçam nos logs do Cloud Run
CMD ["python", "-u", "main.py"]