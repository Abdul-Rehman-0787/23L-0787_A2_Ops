import logging
import os

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Config
from .db import Database
from .routes import AppError, failResponse, maxBytes, router


def buildPredictor():
    log = logging.getLogger("visionguard")
    if Config.INFERENCE_MODE == "remote":
        from .remote import RemotePredictor
        log.info("Inference mode: REMOTE -> %s", Config.API_URL)
        return RemotePredictor(Config.API_URL, Config.API_KEY, Config.API_TIMEOUT)
    from .inference import Predictor
    log.info("Inference mode: LOCAL (%s)", Config.MODEL_PATH)
    return Predictor(Config.MODEL_PATH, Config.THRESHOLD)


def createApp(databaseUrl=None, predictor=None):
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="VisionGuard")
    app.state.config = Config
    app.state.db = Database(databaseUrl or Config.DATABASE_URL)
    app.state.predictor = predictor or buildPredictor()

    staticFolder = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=staticFolder), name="static")
    app.include_router(router)

    @app.middleware("http")
    async def limitUploadSize(request: Request, callNext):
        # Refuse huge uploads before reading them (1 MB extra for the form overhead).
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > maxBytes + 1024 * 1024:
            megabytes = maxBytes // (1024 * 1024)
            return failResponse(request, 413, "File too large",
                                "The file is larger than the " + str(megabytes) + " MB limit.")
        return await callNext(request)

    @app.exception_handler(AppError)
    async def handleAppError(request: Request, error: AppError):
        return failResponse(request, error.code, error.title, error.message)

    @app.exception_handler(StarletteHTTPException)
    async def handleHttpError(request: Request, error: StarletteHTTPException):
        if error.status_code == 404:
            return failResponse(request, 404, "Page not found", "That page doesn't exist.")
        if error.status_code == 405:
            return failResponse(request, 405, "Not allowed", "That action isn't allowed on this page.")
        return failResponse(request, error.status_code, "Something went wrong", "Please try again.")

    @app.exception_handler(Exception)
    async def handleCrash(request: Request, error: Exception):
        logging.getLogger("visionguard").exception("Unhandled error")
        return failResponse(request, 500, "Server error",
                            "Something went wrong on our side. Please try again.")

    return app