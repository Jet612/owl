"""Access tokens for the public API.

The website's server holds OWL_API_SECRET. It can call the API with
`Authorization: Bearer <secret>`, and it mints short-lived media tokens for
browsers, which can't attach headers to <video> or HLS requests:

    token = f"{exp}.{base64url(hmac_sha256(secret, f'owl-media:{exp}'))}"

where `exp` is a Unix timestamp in seconds and base64url has no padding.
"""

import base64
import hashlib
import hmac
import time


def _signature(secret: str, exp: int) -> str:
    digest = hmac.new(secret.encode(), f"owl-media:{exp}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def make_token(secret: str, ttl: int = 6 * 3600) -> str:
    exp = int(time.time()) + ttl
    return f"{exp}.{_signature(secret, exp)}"


def check_token(secret: str, token: str) -> bool:
    exp_text, _, signature = token.partition(".")
    if not exp_text.isdigit() or int(exp_text) < time.time():
        return False
    return hmac.compare_digest(signature, _signature(secret, int(exp_text)))


def check_bearer(secret: str, header: str) -> bool:
    scheme, _, value = header.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(value.strip(), secret)
