"""
EEG Processing Service – FastAPI server
Receives raw EEG chunks from the WPF application, runs preprocessing and model inference,
returns structured results including component waves, DSA, band powers, BIS, and HRV.

Start:  python main.py
or:     uvicorn main:app --host 0.0.0.0 --port 8765 --reload
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Configure structured logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
)
logger.add(
    Path.home() / "EEGMonitor" / "Logs" / "service.log",
    rotation="50 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

from api.router import router, init_services
from api.simulate_router import router as simulate_router

app = FastAPI(
    title="EEG Processing Service",
    description="AnesthesiaNetV2 real-time EEG processing and BIS prediction",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(simulate_router)


@app.on_event("startup")
async def startup():
    # Resolve model path relative to the tianjin project root
    model_path = Path(__file__).resolve().parents[2] / "checkpoints" / "best_model.pth"
    init_services(model_path=str(model_path) if model_path.exists() else None)
    logger.info(f"EEG Processing Service started on http://localhost:8765")
    if not model_path.exists():
        logger.warning(f"No model checkpoint found at {model_path} – using heuristic BIS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Processing Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
