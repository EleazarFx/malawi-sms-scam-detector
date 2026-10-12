"""
FastAPI web app for Malawi SMS Scam Detection.

Run:
    uvicorn app.main:app --reload
Open: http://127.0.0.1:8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_sms  # noqa: E402

# Production threshold — see threshold-tuning analysis in README.
# At 0.4: FN=0 (all scams caught), FP=7. FP is far cheaper than FN.
PRODUCTION_THRESHOLD = 0.4

app = FastAPI(title="Malawi SMS Scam Detector")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "message": ""},
    )


@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, message: str = Form(...)):
    if not message.strip():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"result": None, "message": ""},
        )

    result = predict_sms(message, threshold=PRODUCTION_THRESHOLD)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result, "message": message},
    )


@app.get("/api/predict")
async def api_predict(text: str, threshold: float = PRODUCTION_THRESHOLD):
    """JSON API: /api/predict?text=...&threshold=0.4"""
    return predict_sms(text, threshold=threshold)