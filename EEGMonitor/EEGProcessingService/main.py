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
    description="AnesthesiaNetV3 (MERIDIAN v13) real-time EEG processing and BIS prediction",
    version="3.0.0",
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
    # 部署目标 = v17（shared 头 + 不确定度 + 药物派生相位）。一旦 v17 的 .pt 放入即自动启用。
    # v17 需在 GPU 机训练产出（本机 CPU 无法训练）。回退链按 val_mae 排序，回退时 WARNING。
    root = Path(__file__).resolve().parents[2]   # tianjin/
    candidates = [
        root / "outputs" / "checkpoints" / "v17" / "best_model_v3.pt",  # 目标
        root / "outputs" / "checkpoints" / "v13" / "best_model_v3.pt",  # 回退：MAE 4.57(最优可用)
        root / "outputs" / "checkpoints" / "v14" / "best_model_v3.pt",  # 回退：shared 头(MAE 5.8)
        root / "outputs" / "checkpoints" / "best_model.pt",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    init_services(model_path=str(model_path) if model_path else None)
    logger.info("EEG Processing Service started on http://localhost:8765")
    if model_path:
        logger.info(f"Using checkpoint: {model_path}")
        if "v17" not in str(model_path).replace("\\", "/"):
            logger.warning(
                "未找到 v17 checkpoint —— 当前回退到非目标模型。"
                "请在 GPU 机执行训练并把 outputs/checkpoints/v17/best_model_v3.pt 放入本机。"
            )
    else:
        logger.warning("No model checkpoint found – using heuristic BIS")


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
