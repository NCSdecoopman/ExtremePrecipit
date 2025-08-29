FROM python:3.11-slim

WORKDIR /extremeprecipit

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton projet
COPY . .

# Commande par défaut
CMD ["python", "main.py"]
