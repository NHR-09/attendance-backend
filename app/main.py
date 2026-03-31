"""
Native Attendance Application — FastAPI Backend
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.config import get_settings
from app.routers import auth, attendance, employees, manager, policy

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables and upload dir
    await init_db()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield
    # Shutdown


app = FastAPI(
    title="Native Attendance API",
    description="Face-based attendance system with device binding, location fallback, and manager controls",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter app + web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads for photo serving
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR, check_dir=False), name="uploads")

# Register routers
app.include_router(auth.router)
app.include_router(attendance.router)
app.include_router(employees.router)
app.include_router(manager.router)
app.include_router(policy.router)


@app.get("/")
async def root():
    return {
        "app": "Native Attendance API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
