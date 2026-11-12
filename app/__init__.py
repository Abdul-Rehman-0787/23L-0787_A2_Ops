import logging
import os
import secrets
from urllib.parse import quote

import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import LoginGuard, OtpService
from .config import Config
from .db import Database
from .graph import GraphStore
from .knowledge import KnowledgeBase
from .llm import OllamaLlm
from .routes import AppError, LoginRequired, failResponse, maxBytes, router, wantsJson


def buildPredictor():
    log = logging.getLogger("visionguard")
    if Config.INFERENCE_MODE == "remote":
        from .remote import RemotePredictor
        log.info("Inference mode: REMOTE -> %s", Config.API_URL)
        return RemotePredictor(Config.API_URL, Config.API_KEY, Config.API_TIMEOUT)
    from .inference import Predictor
    log.info("Inference mode: LOCAL (%s)", Config.MODEL_PATH)
    return Predictor(Config.MODEL_PATH, Config.THRESHOLD)

def buildLlm():
    return OllamaLlm(Config.OLLAMA_URL, Config.OLLAMA_MODEL, Config.LLM_TIMEOUT, Config.LLM_MAX_TOKENS)

def pickSecret(value, name):
    if len(value) >= 32:
        return value
    logging.getLogger("visionguard").warning(
        "%s is missing or shorter than 32 characters: using a temporary random key "
        "(sessions end when the server restarts).", name)
    return secrets.token_urlsafe(48)


def queueEmail(toAddress, subject, bodyText):
    """Hands the email to the Celery worker, so the page does not wait for the mail server."""
    from .tasks import sendEmail
    sendEmail.delay(toAddress, subject, bodyText)

def queueGraph(analysis):
    """Hands the lineage update to the Celery worker."""
    from .tasks import recordGraph
    recordGraph.delay(analysis)

def createApp(databaseUrl=None, predictor=None, redisClient=None, mailSender=None, llm=None, knowledge=None, graph=None, graphWriter=None):
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="VisionGuard")
    app.state.config = Config
    app.state.db = Database(databaseUrl or Config.DATABASE_URL)
    app.state.predictor = predictor or buildPredictor()
    app.state.redis = redisClient or redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
    app.state.jwtSecret = pickSecret(Config.JWT_SECRET, "JWT_SECRET")
    app.state.otp = OtpService(app.state.redis, pickSecret(Config.OTP_SECRET, "OTP_SECRET"))
    app.state.loginGuard = LoginGuard(app.state.redis)
    app.state.sendMail = mailSender or queueEmail
    app.state.llm = llm or buildLlm()
    app.state.knowledge = knowledge or KnowledgeBase(Config.CHROMA_PATH, Config.EMBEDDING_CACHE, Config.KNOWLEDGE_FOLDER)
    app.state.graph = graph or GraphStore(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    app.state.recordGraph = graphWriter or queueGraph


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

    @app.exception_handler(LoginRequired)
    async def handleLoginRequired(request: Request, error: LoginRequired):
        if wantsJson(request):
            return JSONResponse({"error": "Please log in to analyse images.",
                                 "title": "Login required"}, status_code=401)
        return RedirectResponse("/login?next=" + quote(request.url.path), status_code=303)

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
