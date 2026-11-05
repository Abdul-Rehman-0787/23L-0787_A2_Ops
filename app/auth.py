import hashlib
import hmac
import secrets
import time
import uuid

import bcrypt
import jwt


class AuthError(Exception):
    """A problem we can show to the user. The message is safe to display."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.message = message
        self.code = code


def cleanEmail(email):
    email = (email or "").strip().lower()
    atPosition = email.find("@")
    message = "Please enter a valid email address."
    if len(email) > 255 or email.count("@") != 1 or atPosition < 1 or " " in email:
        raise AuthError(message)
    domain = email[atPosition + 1:]
    if "." not in domain or domain.startswith(".") or domain.endswith("."):
        raise AuthError(message)
    return email


def checkPassword(password):
    password = password or ""
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if len(password.encode("utf-8")) > 72:
        raise AuthError("Password is too long (72 bytes maximum).")
    return password


def hashPassword(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def passwordMatches(password, passwordHash):
    try:
        return bcrypt.checkpw(password.encode("utf-8"), passwordHash.encode("ascii"))
    except ValueError:
        return False


def makeCode():
    return str(secrets.randbelow(1000000)).zfill(6)      # 6 digits, secure random source


def hashCode(secret, email, code):
    message = (email + ":" + code).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


class OtpService:
    """One-time codes. Only a keyed hash of the code is stored (in Redis)."""

    codeSeconds = 300
    resendSeconds = 60
    maxTries = 5
    lockSeconds = 900

    def __init__(self, redisClient, secret):
        self.redis = redisClient
        self.secret = secret

    def issueCode(self, email):
        if self.redis.exists("otp_lock:" + email):
            raise AuthError("Too many wrong codes. Please try again in 15 minutes.", 429)
        if self.redis.exists("otp_cool:" + email):
            raise AuthError("Please wait a minute before asking for a new code.", 429)
        code = makeCode()
        self.redis.set("otp:" + email, hashCode(self.secret, email, code), ex=self.codeSeconds)
        self.redis.set("otp_cool:" + email, "1", ex=self.resendSeconds)
        self.redis.delete("otp_tries:" + email)
        return code

    def checkCode(self, email, code):
        if self.redis.exists("otp_lock:" + email):
            raise AuthError("Too many wrong codes. Please try again in 15 minutes.", 429)
        saved = self.redis.get("otp:" + email)
        if saved is None:
            raise AuthError("That code has expired. Please log in again.")
        if hmac.compare_digest(saved, hashCode(self.secret, email, (code or "").strip())):
            self.redis.delete("otp_cool:" + email)
            self.redis.delete("otp:" + email)               # one-time use
            self.redis.delete("otp_tries:" + email)
            return True
        tries = self.redis.incr("otp_tries:" + email)
        self.redis.expire("otp_tries:" + email, self.lockSeconds)
        if tries >= self.maxTries:
            self.redis.set("otp_lock:" + email, "1", ex=self.lockSeconds)
            self.redis.delete("otp:" + email)
        raise AuthError("That code is not correct.")


def makeToken(secret, userId, minutes):
    now = int(time.time())
    payload = {"sub": str(userId), "jti": uuid.uuid4().hex, "iat": now, "exp": now + minutes * 60}
    return jwt.encode(payload, secret, algorithm="HS256")


def readToken(secret, token, redisClient):
    """Returns the token's payload, or None if it is invalid, expired or logged out."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if redisClient.exists("revoked:" + payload["jti"]):
        return None
    return payload


def blockToken(redisClient, payload):
    secondsLeft = int(payload["exp"] - time.time())
    if secondsLeft > 0:
        redisClient.set("revoked:" + payload["jti"], "1", ex=secondsLeft)


def tooManyRequests(redisClient, name, limit, seconds):
    key = "rate:" + name
    count = redisClient.incr(key)
    if count == 1:
        redisClient.expire(key, seconds)
    return count > limit