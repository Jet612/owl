#!/usr/bin/env bash
# Install the owl-vision and owl-api services on the Pi.
#
# Usage (from the repo root, on the Pi, after install-mediamtx.sh):
#   sudo ./scripts/install-owl.sh
#
# Creates /etc/owl/owl.env with a fresh API secret and ntfy topic the first
# time, installs the species list to /etc/owl/species.txt, and downloads the
# species identification model (about 600 MB). Later runs keep your settings
# and species list. Rerunning is safe.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH=/etc/owl/owl.env
SERVICE_USER="${SUDO_USER:-jet}"

if [[ $EUID -ne 0 ]]; then
  echo "run this with sudo" >&2
  exit 1
fi

if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
  echo "Creating the Python environment"
  sudo -u "$SERVICE_USER" bash -c "cd '$REPO_DIR' && uv sync"
fi

install -d -m 0750 -o root -g "$SERVICE_USER" /etc/owl
if [[ ! -f "$ENV_PATH" ]]; then
  secret="$(openssl rand -hex 32)"
  topic="owl-$(openssl rand -hex 8)"
  sed -e "s|^OWL_API_SECRET=.*|OWL_API_SECRET=$secret|" \
      -e "s|^OWL_NTFY_TOPIC=.*|OWL_NTFY_TOPIC=$topic|" \
      "$REPO_DIR/owl.env.example" > "$ENV_PATH"
  chown root:"$SERVICE_USER" "$ENV_PATH"
  chmod 0640 "$ENV_PATH"
  echo "Created $ENV_PATH"
fi

SPECIES_PATH=/etc/owl/species.txt
if [[ ! -f "$SPECIES_PATH" ]]; then
  install -m 0644 "$REPO_DIR/species.txt" "$SPECIES_PATH"
  echo "Installed $SPECIES_PATH"
elif ! cmp -s "$REPO_DIR/species.txt" "$SPECIES_PATH"; then
  install -m 0644 "$REPO_DIR/species.txt" "$SPECIES_PATH.new"
  echo "Kept your edited $SPECIES_PATH; the repo version is at $SPECIES_PATH.new"
fi

# The species model lives in the service's state directory, which must exist and
# belong to the service user before the download.
install -d -m 0750 -o "$SERVICE_USER" -g "$SERVICE_USER" /var/lib/owl /var/lib/owl/models
echo "Fetching the species identification model (first run only)"
sudo -u "$SERVICE_USER" env HF_HOME=/var/lib/owl/models bash -c \
  "cd '$REPO_DIR' && .venv/bin/python -m owl.classify --download" 2>&1 | grep -E "ready|Error|error" || true

for unit in owl-vision owl-api; do
  sed -e "s|/home/jet/owl|$REPO_DIR|g" -e "s|^User=jet|User=$SERVICE_USER|" \
      -e "s|^Group=jet|Group=$SERVICE_USER|" \
      "$REPO_DIR/systemd/$unit.service" > "/etc/systemd/system/$unit.service"
done
systemctl daemon-reload
systemctl enable owl-vision.service owl-api.service >/dev/null
systemctl restart owl-vision.service owl-api.service

# shellcheck disable=SC1090
source "$ENV_PATH"
echo
echo "owl-vision and owl-api are running."
echo
echo "Phone notifications: install the ntfy app and subscribe to topic"
echo "  $OWL_NTFY_TOPIC   (server ${OWL_NTFY_SERVER:-https://ntfy.sh})"
echo
echo "Expose the API publicly (once):"
echo "  sudo tailscale funnel --bg ${OWL_API_PORT:-8080}"
echo
echo "Website server settings: OWL_API_URL=https://$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo '<pi>.ts.net')"
echo "  and OWL_API_SECRET from $ENV_PATH"
