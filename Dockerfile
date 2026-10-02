version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: aqua_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_storage:/root/.ollama
    restart: always

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: aqua_backend
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ./backend/chroma_db:/app/chroma_db
      - ./backend/ml/models:/app/ml/models
    depends_on:
      - ollama
    restart: always

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: aqua_frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: always

volumes:
  ollama_storage: