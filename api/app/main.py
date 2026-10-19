from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import health, prediction
from app.services.prediction_service import prediction_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # Startup - run once when API starts
    prediction_service.load_artifacts()
    print(f"✅ Artifacts loaded. Model ready.")
    yield
    
    # Shutdown - runs once when API stops
    print("🛑 Shutting down.")
    

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])