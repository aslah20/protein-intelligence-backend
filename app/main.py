from fastapi import FastAPI
from app.routes import localization, stability

app = FastAPI(
    title="Protein Intelligence API",
    description="Multi-task Deep Learning API for Protein Analysis",
    version="1.0.0"
)

app.include_router(localization.router, prefix="/predict/localization", tags=["Localization"])
app.include_router(stability.router, prefix="/predict/stability", tags=["Stability"])