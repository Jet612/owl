"""Phone notifications through ntfy (https://ntfy.sh). Every phone subscribed to the topic gets them."""

import logging
import threading
import urllib.request
from pathlib import Path

LOG = logging.getLogger(__name__)


class Notifier:
    def __init__(self, server: str, topic: str, token: str, click_url: str):
        self._url = f"{server.rstrip('/')}/{topic}" if topic else ""
        self._token = token
        self._click_url = click_url
        if not self._url:
            LOG.warning("OWL_NTFY_TOPIC is not set; phone notifications are off")
        elif not click_url:
            LOG.warning("OWL_SITE_URL is not set; tapping a notification will not open the website")

    def send(
        self, title: str, message: str, snapshot: Path | None = None,
        tags: str = "paw_prints", priority: str = "high",
    ) -> None:
        """Send in the background so a slow network never stalls the camera loop."""
        if self._url:
            threading.Thread(
                target=self._send, args=(title, message, snapshot, tags, priority), daemon=True
            ).start()

    def _send(self, title: str, message: str, snapshot: Path | None, tags: str, priority: str) -> None:
        headers = {"Title": title, "Tags": tags, "Priority": priority}
        if self._click_url:
            headers["Click"] = self._click_url
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if snapshot and snapshot.exists():
            # ntfy takes the attachment as the body and the text in a header.
            body = snapshot.read_bytes()
            headers["Filename"] = snapshot.name
            headers["Message"] = message
        else:
            body = message.encode()
        request = urllib.request.Request(self._url, data=body, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(request, timeout=20):
                pass
            LOG.info("Sent notification: %s", title)
        except Exception as exc:
            LOG.warning("Notification failed: %s", exc)
