from fastapi import FastAPI

app = FastAPI(title="Aqua XAI API")

@app.get("/")
def read_root():
    return {"status": "online", "system": "Aqua XAI Backend"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
