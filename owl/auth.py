"""Access tokens for the public API.

The website's server holds OWL_API_SECRET. It can call the API with
`Authorization: Bearer <secret>`, and it mints short-lived media tokens for
browsers, which can't attach headers to <video> or HLS requests:

    token = f"{exp}.{base64url(hmac_sha256(secret, f'owl-media:{exp}'))}"

where `exp` is a Unix timestamp in seconds and base64url has no padding.

Nothing here raises on bad input: a credential is either valid or it isn't.
"""

import base64
import hashlib
import hmac
import re
import time

TOKEN = re.compile(r"([0-9]{1,12})\.([A-Za-z0-9_-]{1,64})")


def _signature(secret: str, exp: str) -> str:
    digest = hmac.new(secret.encode(), f"owl-media:{exp}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def make_token(secret: str, ttl: int = 6 * 3600) -> str:
    exp = int(time.time()) + ttl
    return f"{exp}.{_signature(secret, str(exp))}"


def check_token(secret: str, token: str) -> bool:
    match = TOKEN.fullmatch(token)
    if not secret or not match or int(match[1]) <= time.time():
        return False
    return hmac.compare_digest(_signature(secret, match[1]).encode(), match[2].encode())


def check_bearer(secret: str, header: str) -> bool:
    scheme, _, value = header.partition(" ")
    if not secret or scheme.lower() != "bearer":
        return False
    # aiohttp decodes header bytes with surrogateescape, so encode them back the same way.
    return hmac.compare_digest(value.strip().encode("utf-8", "surrogateescape"), secret.encode())
