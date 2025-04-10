# Utiliser l'image officielle Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn[standard]

# Copier le code de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'application FastAPI écoute
EXPOSE 7890

# Commande pour lancer l'application avec uvicorn
CMD ["uvicorn", "server_lite:app", "--host", "0.0.0.0", "--port", "7890"]