import pytest

from app.auth import (
    AuthError,
    OtpService,
    blockToken,
    checkPassword,
    cleanEmail,
    hashPassword,
    makeToken,
    passwordMatches,
    readToken,
    tooManyRequests,
)


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value

    def exists(self, key):
        return 1 if key in self.values else 0

    def delete(self, key):
        self.values.pop(key, None)

    def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    def expire(self, key, seconds):
        pass


def makeOtp():
    return OtpService(FakeRedis(), "test-secret"), "a@b.com"


def testCleanEmail():
    assert cleanEmail("  Abdul@Example.COM ") == "abdul@example.com"
    for bad in ["", "no-at-sign", "a@b", "a@@b.com", "@b.com", "a b@c.com", "a@.com"]:
        with pytest.raises(AuthError):
            cleanEmail(bad)


def testPasswordRules():
    assert checkPassword("longenough") == "longenough"
    with pytest.raises(AuthError):
        checkPassword("short")
    with pytest.raises(AuthError):
        checkPassword("x" * 73)


def testPasswordHash():
    hashed = hashPassword("my password 1")
    assert hashed != "my password 1"
    assert passwordMatches("my password 1", hashed)
    assert not passwordMatches("wrong password", hashed)


def testOtpCodeWorksOnceAndIsNotStoredAsPlainText():
    otp, email = makeOtp()
    code = otp.issueCode(email)
    assert code not in otp.redis.values.values()
    assert otp.checkCode(email, code) is True
    with pytest.raises(AuthError):
        otp.checkCode(email, code)


def testOtpLocksAfterFiveWrongTries():
    otp, email = makeOtp()
    code = otp.issueCode(email)
    for _ in range(5):
        with pytest.raises(AuthError):
            otp.checkCode(email, "000000" if code != "000000" else "111111")
    with pytest.raises(AuthError) as error:
        otp.checkCode(email, code)
    assert error.value.code == 429


def testOtpResendCooldown():
    otp, email = makeOtp()
    otp.issueCode(email)
    with pytest.raises(AuthError) as error:
        otp.issueCode(email)
    assert error.value.code == 429


def testTokenRoundTripAndLogout():
    redisClient = FakeRedis()
    token = makeToken("secret-one", 7, 60)
    payload = readToken("secret-one", token, redisClient)
    assert payload["sub"] == "7"
    assert readToken("another-secret", token, redisClient) is None
    blockToken(redisClient, payload)
    assert readToken("secret-one", token, redisClient) is None


def testExpiredTokenIsRejected():
    token = makeToken("secret-one", 7, -1)
    assert readToken("secret-one", token, FakeRedis()) is None


def testRateLimit():
    redisClient = FakeRedis()
    answers = [tooManyRequests(redisClient, "ip", 3, 60) for _ in range(5)]
    assert answers == [False, False, False, True, True]