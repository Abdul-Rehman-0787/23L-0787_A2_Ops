import base64
import hashlib
import io
import logging
import os
import time
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from .config import Config
from .icons import icon

router = APIRouter()
log = logging.getLogger("visionguard")

allowedExtensions = [".png", ".jpg", ".jpeg"]
allowedFormats = ["JPEG", "PNG"]
maxPixels = 20_000_000          # same limit as the model API
maxBytes = Config.MAX_CONTENT_LENGTH

templateFolder = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templateFolder)
templates.env.globals["icon"] = icon
templates.env.globals["config"] = Config

# Your templates were written for Flask. These two small helpers keep them working.
pathToEndpoint = {"/": "main.index", "/predict": "main.analyze",
                  "/history": "main.history", "/health": "main.health"}


def flaskUrlFor(endpoint, **values):
    if endpoint == "static":
        return "/static/" + str(values.get("filename", ""))
    for path, mapped_endpoint in pathToEndpoint.items():
        if mapped_endpoint == endpoint:
            return path
    return "/"


templates.env.globals["url_for"] = flaskUrlFor


class AppError(Exception):
    """An error we want to show to the user as a friendly message."""

    def __init__(self, code, title, message):
        self.code = code
        self.title = title
        self.message = message


def wantsJson(request):
    return request.headers.get("x-requested-with") == "fetch"


def render(request, pageName, context, code=200):
    request.path = request.url.path                      # Flask-style names for the templates
    request.endpoint = pathToEndpoint.get(request.url.path, "")
    return templates.TemplateResponse(request, pageName, context, status_code=code)


def failResponse(request, code, title, message):
    """Friendly error: JSON for the page's JavaScript, an HTML page otherwise."""
    if wantsJson(request):
        return JSONResponse({"error": message, "title": title}, status_code=code)
    return render(request, "error.html", {"code": code, "title": title, "message": message}, code)


def cleanFilename(name):
    """Keep only the file name (no folders) and safe characters."""
    name = name.replace("\\", "/").split("/")[-1]
    cleaned = ""
    for character in name:
        if character.isalnum() or character in "._-":
            cleaned = cleaned + character
        else:
            cleaned = cleaned + "_"
    return cleaned.strip("._")[:255]


def checkUpload(fileName, data):
    title = "We couldn't use that file"
    if not fileName:
        raise AppError(400, title, "No file was selected. Please choose an image.")
    extension = os.path.splitext(fileName)[1].lower()
    if extension not in allowedExtensions:
        raise AppError(400, title, "Unsupported file type. Please upload a JPG or PNG image.")
    safeName = cleanFilename(fileName) or "upload" + extension
    if len(data) == 0:
        raise AppError(400, title, "The uploaded file is empty.")
    if len(data) > maxBytes:
        megabytes = maxBytes // (1024 * 1024)
        raise AppError(413, "File too large", "The file is larger than the " + str(megabytes) + " MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            imageFormat = image.format
            width, height = image.size
            image.verify()
    except (OSError, SyntaxError, ValueError):
        raise AppError(400, title, "This file looks like an image but could not be decoded. "
                                   "It may be corrupt or not really an image.") from None
    if imageFormat not in allowedFormats:
        raise AppError(400, title, "The file content is not a JPG or PNG image.")
    if width * height > maxPixels:
        raise AppError(400, title, "The image has too many pixels. Please upload a smaller one.")
    return safeName, width, height


def thumbnailUri(data):
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        image.thumbnail((480, 480))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@router.get("/")
def index(request: Request):
    return render(request, "index.html", {})


@router.post("/predict")
async def analyze(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
):
    fileName = ""
    data = b""
    if image is not None:
        fileName = image.filename or ""
        data = await image.read(maxBytes + 1)            # stops reading just past the limit
    safeName, width, height = checkUpload(fileName, data)
    imageHash = hashlib.sha256(data).hexdigest()

    startTime = time.perf_counter()
    try:
        result = await run_in_threadpool(request.app.state.predictor.predict, data)
    except Exception:
        log.exception("Inference failed")
        raise AppError(500, "Analysis failed",
                       "Something went wrong while analysing the image.") from None
    milliseconds = (time.perf_counter() - startTime) * 1000

    config = request.app.state.config
    saved = True
    try:
        await run_in_threadpool(
            request.app.state.db.add_prediction,
            filename=safeName[:255], image_sha256=imageHash,
            label=result["label"], confidence=result["confidence"],
            prob_ai=result["prob_ai"], threshold=result["threshold"],
            model_name=result.get("model_name") or config.MODEL_NAME,
            model_version=str(result.get("model_version") or config.MODEL_VERSION),
            width=width, height=height, file_size=len(data), latency_ms=milliseconds)
    except Exception:
        log.exception("Could not write to the database")
        saved = False

    if wantsJson(request):
        return JSONResponse({
            "label": result["label"], "confidence": result["confidence"],
            "prob_ai": result["prob_ai"], "threshold": result["threshold"],
            "model": config.MODEL_NAME, "model_version": str(config.MODEL_VERSION),
            "filename": safeName, "sha256": imageHash, "width": width, "height": height,
            "file_size": len(data), "latency_ms": round(milliseconds, 1), "saved": saved})
    return render(request, "result.html", {
        "res": result, "name": safeName, "w": width, "h": height, "saved": saved,
        "preview": thumbnailUri(data), "model": config.MODEL_NAME,
        "version": config.MODEL_VERSION})


@router.get("/history")
def history(request: Request):
    rows = request.app.state.db.recent(100)
    total = len(rows)
    aiCount = 0
    confidenceSum = 0.0
    for row in rows:
        if row.label == "AI-generated":
            aiCount = aiCount + 1
        confidenceSum = confidenceSum + row.confidence
    average = 0
    if total > 0:
        average = confidenceSum / total * 100
    stats = {"total": total, "ai": aiCount, "real": total - aiCount, "avg": average}
    return render(request, "history.html", {"rows": rows, "stats": stats})


@router.get("/health")
def health(request: Request):
    databaseOk = request.app.state.db.ping()
    config = request.app.state.config
    body = {"status": "ok" if databaseOk else "degraded", "model_loaded": True,
            "model": config.MODEL_NAME, "model_version": str(config.MODEL_VERSION),
            "threshold": config.THRESHOLD, "database": "ok" if databaseOk else "error"}
    return JSONResponse(body, status_code=200 if databaseOk else 503)