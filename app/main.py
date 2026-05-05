from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.services.assistant_service import process_chat

app = FastAPI(
    title="ArabSoft AI Service",
    description="Local rule-based HR assistant adapter for the ArabSoft PFE platform.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to ArabSoft AI Service",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "UP", "service": "arabsoft-ai-service"}


@app.post("/assistant/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return process_chat(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
