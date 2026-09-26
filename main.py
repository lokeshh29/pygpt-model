from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="PyGPT Model API",
    description="A basic FastAPI server configured with uv",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI server!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
