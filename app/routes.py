import base64
import hashlib
import io
import json
import logging
import os
import time
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from redis.exceptions import RedisError

from .artist import buildMessages, maxQuestionLength
from .auth import (
    AuthError,
    blockToken,
    checkPassword,
    cleanEmail,
    hashPassword,
    makeToken,
    passwordMatches,
    readToken,
    tooManyRequests,
)
from .config import Config
from .icons import icon
from .llm import LlmError

router = APIRouter()
log = logging.getLogger("visionguard")

allowedExtensions = [".png", ".jpg", ".jpeg"]
allowedFormats = ["JPEG", "PNG"]
maxPixels = 20_000_000          # same limit as the model API
maxBytes = Config.MAX_CONTENT_LENGTH
cookieName = "vg_token"

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

# Used when the email does not exist, so a wrong email and a wrong password take equally long.
dummyHash = hashPassword("not-a-real-password")


class AppError(Exception):
    """An error we want to show to the user as a friendly message."""

    def __init__(self, code, title, message):
        self.code = code
        self.title = title
        self.message = message


class LoginRequired(Exception):
    """Raised when a page needs a logged-in user."""


def wantsJson(request):
    return request.headers.get("x-requested-with") == "fetch"


def clientIp(request):
    if request.client is None:
        return ""
    return request.client.host


def safeNext(value):
    """Only allow a redirect to a page on this site."""
    if value.startswith("/") and not value.startswith("//") and "\\" not in value:
        return value
    return "/"


def getUser(request):
    """Returns the logged-in user, or None."""
    token = request.cookies.get(cookieName, "")
    if token == "":
        return None
    payload = readToken(request.app.state.jwtSecret, token, request.app.state.redis)
    if payload is None:
        return None
    user = request.app.state.db.findUserById(int(payload["sub"]))
    if user is None or not user.is_active:
        return None
    request.state.tokenPayload = payload
    return user


def safeUser(request):
    """Like getUser, but a Redis problem never breaks a page."""
    try:
        return getUser(request)
    except Exception:
        log.exception("Could not check the login")
        return None


def requireUser(request: Request):
    user = getUser(request)
    if user is None:
        raise LoginRequired()
    return user


def render(request, pageName, context, code=200):
    request.path = request.url.path                      # Flask-style names for the templates
    request.endpoint = pathToEndpoint.get(request.url.path, "")
    context["user"] = safeUser(request)
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


def limitAttempts(request, key, limit, seconds):
    if tooManyRequests(request.app.state.redis, key, limit, seconds):
        raise AuthError("Too many attempts. Please wait a few minutes and try again.", 429)


def sendCode(request, email, userId):
    """Make a one-time code and queue the email that carries it."""
    code = request.app.state.otp.issueCode(email)
    if Config.DEV_LOG_CODES:
        log.info("DEV ONLY: the code for %s is %s", email, code)
    text = ("Your VisionGuard code is " + code + ". It expires in 5 minutes.\n"
            "If you did not ask for it, you can ignore this email.")
    try:
        request.app.state.sendMail(email, "Your VisionGuard login code", text)
    except Exception:
        log.exception("Could not queue the email")
        raise AuthError("We could not send the email right now. "
                        "Please try again in a minute.", 503) from None
    request.app.state.db.addEvent("otp_sent", userId, clientIp(request))


def sendToGraph(request, userId, row, fileName, imageHash, result):
    """Queue the lineage update. A graph problem must never break an analysis."""
    if not request.app.state.graph.enabled:
        return
    config = request.app.state.config
    analysis = {
        "analysisId": row["id"], "userId": userId, "sha256": imageHash,
        "filename": fileName[:40], "label": result["label"],
        "probAi": float(result["prob_ai"]), "threshold": float(result["threshold"]),
        "createdAt": row["created_at"].isoformat(),
        "modelName": result.get("model_name") or config.MODEL_NAME,
        "modelVersion": str(result.get("model_version") or config.MODEL_VERSION),
    }
    try:
        request.app.state.recordGraph(analysis)
    except Exception:
        log.exception("Could not queue the graph update")




@router.get("/")
def index(request: Request):
    return render(request, "index.html", {})


# ---------------------------------------------------------------- accounts

@router.get("/register")
def registerPage(request: Request):
    return render(request, "register.html", {"error": "", "email": ""})


@router.post("/register")
def registerSubmit(request: Request, email: str = Form(""), password: str = Form("")):
    db = request.app.state.db
    try:
        limitAttempts(request, "register-ip:" + clientIp(request), 10, 300)
        email = cleanEmail(email)
        password = checkPassword(password)
        hashed = hashPassword(password)
        user = db.createUser(email, hashed)
        if user is None:
            user = db.findUserByEmail(email)
            if user.is_verified:
                raise AuthError("An account with this email already exists. Please log in.")
            db.updatePassword(user.id, hashed)       # never verified: the new owner may take it
        db.addEvent("register", user.id, clientIp(request))
        sendCode(request, email, user.id)
    except AuthError as error:
        return render(request, "register.html", {"error": error.message, "email": email}, error.code)
    return RedirectResponse("/verify?email=" + quote(email), status_code=303)


@router.get("/login")
def loginPage(request: Request):
    nextPage = safeNext(request.query_params.get("next", "/"))
    return render(request, "login.html", {"error": "", "email": "", "next": nextPage})


@router.post("/login")
def loginSubmit(request: Request, email: str = Form(""), password: str = Form(""),
                nextPage: str = Form("/", alias="next")):
    db = request.app.state.db
    guard = request.app.state.loginGuard
    nextPage = safeNext(nextPage)
    try:
        limitAttempts(request, "login-ip:" + clientIp(request), 10, 300)
        email = cleanEmail(email)
        guard.checkNotLocked(email)
        user = db.findUserByEmail(email)
        passwordIsRight = False
        if user is not None and user.is_active:
            passwordIsRight = passwordMatches(password, user.password_hash)
        else:
            passwordMatches(password, dummyHash)     # same work whether or not the email exists
        if not passwordIsRight:
            userId = None
            if user is not None:
                userId = user.id
            db.addEvent("login_failed", userId, clientIp(request))
            if guard.recordFailure(email):
                db.addEvent("login_locked", userId, clientIp(request))
                raise AuthError("Too many wrong passwords. Please try again in 15 minutes.", 429)
            raise AuthError("Email or password is incorrect.", 401)
        guard.recordSuccess(email)
        sendCode(request, email, user.id)
    except AuthError as error:
        context = {"error": error.message, "email": email, "next": nextPage}
        return render(request, "login.html", context, error.code)
    return RedirectResponse("/verify?email=" + quote(email) + "&next=" + quote(nextPage),
                            status_code=303)


@router.get("/verify")
def verifyPage(request: Request):
    email = request.query_params.get("email", "").strip().lower()
    nextPage = safeNext(request.query_params.get("next", "/"))
    return render(request, "verify.html", {"error": "", "email": email, "next": nextPage})


@router.post("/verify")
def verifySubmit(request: Request, email: str = Form(""), code: str = Form(""),
                 nextPage: str = Form("/", alias="next")):
    db = request.app.state.db
    nextPage = safeNext(nextPage)
    context = {"error": "", "email": email, "next": nextPage}
    try:
        limitAttempts(request, "verify-ip:" + clientIp(request), 20, 300)
        email = cleanEmail(email)
        context["email"] = email
        request.app.state.otp.checkCode(email, code)
    except AuthError as error:
        db.addEvent("otp_failed", None, clientIp(request))
        context["error"] = error.message
        return render(request, "verify.html", context, error.code)

    user = db.findUserByEmail(email)
    if user is None or not user.is_active:
        context["error"] = "This account cannot log in."
        return render(request, "verify.html", context, 400)
    db.recordLogin(user.id)
    db.addEvent("login_ok", user.id, clientIp(request))
    token = makeToken(request.app.state.jwtSecret, user.id, Config.JWT_MINUTES)
    response = RedirectResponse(nextPage, status_code=303)
    response.set_cookie(cookieName, token, max_age=Config.JWT_MINUTES * 60, httponly=True,
                        samesite="lax", secure=Config.COOKIE_SECURE)
    return response


@router.post("/logout")
def logout(request: Request):
    user = safeUser(request)
    payload = getattr(request.state, "tokenPayload", None)
    if user is not None and payload is not None:
        blockToken(request.app.state.redis, payload)
        request.app.state.db.addEvent("logout", user.id, clientIp(request))
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(cookieName)
    return response


# ---------------------------------------------------------------- the product

@router.post("/predict")
async def analyze(
    request: Request,
    user: Annotated[object, Depends(requireUser)],
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
    savedRow = None
    try:
        savedRow = await run_in_threadpool(
            request.app.state.db.add_prediction,
            user_id=user.id, filename=safeName[:255], image_sha256=imageHash,
            label=result["label"], confidence=result["confidence"],
            prob_ai=result["prob_ai"], threshold=result["threshold"],
            model_name=result.get("model_name") or config.MODEL_NAME,
            model_version=str(result.get("model_version") or config.MODEL_VERSION),
            width=width, height=height, file_size=len(data), latency_ms=milliseconds)
    except Exception:
        log.exception("Could not write to the database")
        saved = False
    if savedRow is not None:
        await run_in_threadpool(sendToGraph, request, user.id, savedRow, safeName, imageHash, result)

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
def history(request: Request, user: Annotated[object, Depends(requireUser)]):
    rows = request.app.state.db.recent(100, userId=user.id)
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

@router.get("/artist")
def artistPage(request: Request, user: Annotated[object, Depends(requireUser)]):
    saved = request.app.state.db.recentChat(user.id, 30)
    shown = []
    for message in saved:
        sources = []
        if message.sources:
            try:
                sources = json.loads(message.sources)
            except ValueError:
                sources = []
        shown.append({"role": message.role, "content": message.content, "sources": sources})
    return render(request, "artist.html", {"messages": shown, "maxLength": maxQuestionLength})


async def findPassages(request, question):
    """The best document passages for a question. Artist still answers if the search breaks."""
    try:
        found = await run_in_threadpool(request.app.state.knowledge.search, question, 3)
    except Exception:
        log.exception("Knowledge search failed")
        return []
    good = []
    for hit in found:
        if hit["score"] >= Config.RAG_MIN_SIMILARITY and len(good) < Config.RAG_PROMPT_PASSAGES:
            good.append(hit)
    return good


@router.post("/artist/ask")
async def artistAsk(request: Request, user: Annotated[object, Depends(requireUser)]):
    try:
        body = await request.json()
    except ValueError:
        body = {}
    question = ""
    if isinstance(body, dict):
        question = str(body.get("question", "")).strip()
    if question == "":
        raise AppError(400, "Empty question", "Please type a question for Artist.")
    if len(question) > maxQuestionLength:
        raise AppError(400, "Question too long",
                       "Please keep your question under " + str(maxQuestionLength) + " characters.")
    if tooManyRequests(request.app.state.redis, "artist-user:" + str(user.id), 20, 600):
        raise AppError(429, "Slow down", "You have asked many questions. Please wait a few minutes.")

    db = request.app.state.db
    scans = await run_in_threadpool(db.recent, 5, user.id)
    memory = await run_in_threadpool(db.recentChat, user.id, 4)
    hits = await findPassages(request, question)
    messages = buildMessages(scans, memory, question, hits)

    startTime = time.perf_counter()
    try:
        answer = await run_in_threadpool(request.app.state.llm.chat, messages)
    except LlmError as error:
        raise AppError(503, "Artist is offline", str(error)) from None
    seconds = round(time.perf_counter() - startTime, 1)

    sources = []
    number = 1
    for hit in hits:
        sources.append({"n": number, "source": hit["source"], "section": hit["section"],
                        "snippet": hit["text"][:160], "score": hit["score"]})
        number = number + 1
    await run_in_threadpool(db.addChat, user.id, "user", question)
    await run_in_threadpool(db.addChat, user.id, "assistant", answer, json.dumps(sources))
    log.info("Artist answered in %.1f s using %d passages", seconds, len(sources))
    return JSONResponse({"answer": answer, "sources": sources, "seconds": seconds})
    

@router.post("/artist/clear")
def artistClear(request: Request, user: Annotated[object, Depends(requireUser)]):
    request.app.state.db.deleteChat(user.id)
    return RedirectResponse("/artist", status_code=303)


def redisIsUp(request):
    try:
        return bool(request.app.state.redis.ping())
    except (AttributeError, RedisError, TypeError, ValueError):
        return False


@router.get("/health")
def health(request: Request):
    databaseOk = request.app.state.db.ping()
    config = request.app.state.config
    body = {"status": "ok" if databaseOk else "degraded", "model_loaded": True,
            "model": config.MODEL_NAME, "model_version": str(config.MODEL_VERSION),
            "threshold": config.THRESHOLD, "database": "ok" if databaseOk else "error",
            "redis": "ok" if redisIsUp(request) else "error"}
    return JSONResponse(body, status_code=200 if databaseOk else 503)