from fastapi import FastAPI
import uvicorn

from app.routers import model_router, tokenizer_router

app = FastAPI(
    title="PyGPT Model API",
    description="A modular FastAPI server serving the PyGPT Model specifications, tokenizer, and AI capabilities.",
    version="0.1.0",
)

# Include modular API routers
app.include_router(model_router)
app.include_router(tokenizer_router)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to PyGPT Model FastAPI server!"}


@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
