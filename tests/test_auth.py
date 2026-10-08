"""Token and secret checks, including the website's own mintOwlToken (run under Node)."""

import os
import shutil
import subprocess
import time
import unittest

from owl import auth

SECRET = "a-very-secret-string-of-at-least-32-characters"

# Copied from the website spec; it must keep matching what the site mints.
MINT_OWL_TOKEN = """
import crypto from "node:crypto";

export function mintOwlToken(secret, ttlSeconds = 6 * 3600) {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const sig = crypto.createHmac("sha256", secret).update(`owl-media:${exp}`).digest("base64url");
  return `${exp}.${sig}`;
}
console.log(mintOwlToken(process.env.OWL_TEST_SECRET, Number(process.env.OWL_TEST_TTL)));
"""


def mint_with_node(secret: str, ttl: int) -> str:
    env = {**os.environ, "OWL_TEST_SECRET": secret, "OWL_TEST_TTL": str(ttl)}
    done = subprocess.run(
        ["node", "--input-type=module", "-e", MINT_OWL_TOKEN],
        env=env, capture_output=True, text=True, check=True,
    )
    return done.stdout.strip()


class TokenTest(unittest.TestCase):
    def test_a_fresh_token_is_accepted(self):
        self.assertTrue(auth.check_token(SECRET, auth.make_token(SECRET)))

    def test_an_expired_token_is_rejected(self):
        self.assertFalse(auth.check_token(SECRET, auth.make_token(SECRET, ttl=-1)))

    def test_a_token_that_expires_this_second_is_rejected(self):
        exp = str(int(time.time()))
        self.assertFalse(auth.check_token(SECRET, f"{exp}.{auth._signature(SECRET, exp)}"))

    def test_a_changed_signature_or_expiry_is_rejected(self):
        exp, sig = auth.make_token(SECRET).split(".")
        flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
        self.assertFalse(auth.check_token(SECRET, f"{exp}.{flipped}"))
        self.assertFalse(auth.check_token(SECRET, f"{int(exp) + 1}.{sig}"))

    def test_the_wrong_secret_is_rejected(self):
        self.assertFalse(auth.check_token(SECRET + "x", auth.make_token(SECRET)))

    def test_malformed_tokens_are_rejected_without_raising(self):
        good = auth.make_token(SECRET)
        exp, sig = good.split(".")
        malformed = [
            "", ".", "abc", "123", "123.", ".abc", f"{exp}", f"{exp}.", f"{exp}..{sig}",
            f"{exp}.{sig}.extra", f" {good}", f"{good} ", f"{good}\n", f"{exp}.{sig}=",
            "9" * 13 + f".{sig}",          # more digits than the format allows
            "9" * 5000 + f".{sig}",        # past Python's int/str conversion limit
            f"{exp}.{'a' * 65}",           # signature too long
            "²." + sig, "٣" + exp + f".{sig}",  # digits that are not 0-9
            f"{exp}.é", f"{exp}.{sig}é", "9999999999.\udcff",  # not ASCII
            f"-{exp}.{sig}", f"+{exp}.{sig}", f"{exp}.{sig}\x00",
        ]
        for token in malformed:
            with self.subTest(token=token[:40]):
                self.assertFalse(auth.check_token(SECRET, token))

    def test_an_empty_secret_never_authenticates(self):
        self.assertFalse(auth.check_token("", auth.make_token("")))
        self.assertFalse(auth.check_bearer("", "Bearer "))
        self.assertFalse(auth.check_bearer("", "Bearer"))

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_tokens_from_the_websites_mint_function(self):
        self.assertTrue(auth.check_token(SECRET, mint_with_node(SECRET, 6 * 3600)))
        self.assertFalse(auth.check_token(SECRET, mint_with_node(SECRET, -10)))
        self.assertFalse(auth.check_token(SECRET, mint_with_node("another-secret" * 4, 3600)))


class BearerTest(unittest.TestCase):
    def test_the_secret_is_accepted(self):
        self.assertTrue(auth.check_bearer(SECRET, f"Bearer {SECRET}"))
        self.assertTrue(auth.check_bearer(SECRET, f"bearer {SECRET}  "))

    def test_anything_else_is_rejected_without_raising(self):
        for header in ("", "Bearer", "Bearer ", f"Basic {SECRET}", SECRET, f"Bearer {SECRET}x",
                       "Bearer wrong", "Bearer é", "Bearer \udcff\udcfe"):
            with self.subTest(header=header):
                self.assertFalse(auth.check_bearer(SECRET, header))


if __name__ == "__main__":
    unittest.main()
