from fastapi import FastAPI
from app.routes import localization#, stability, immunogenicity, binding, ptm

app = FastAPI(
    title="Protein Intelligence API",
    description="Multi-task Deep Learning API for Protein Analysis",
    version="1.0.0"
)

app.include_router(localization.router, tags=["Localization"])
# app.include_router(stability.router, tags=["Stability"])  #prefix="/predict/stability",
# app.include_router(immunogenicity.router, tags=["Immunogenicity"])
# app.include_router(binding.router, tags=["Ligand-Binding"])
# app.include_router(ptm.router, tags=["PTM"])

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)