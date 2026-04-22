from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Enterprise Copilot API")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from app.config import Config
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)