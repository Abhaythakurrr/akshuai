# akshu-ai/services/execution/main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Execution service is running"}

# Add endpoints for executing tasks (e.g., running code, interacting with tools)