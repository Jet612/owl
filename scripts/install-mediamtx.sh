#!/usr/bin/env bash
# Install MediaMTX on a Raspberry Pi and run it as a systemd service.
#
# Usage (from the repo root, on the Pi):
#   sudo ./scripts/install-mediamtx.sh            # install or upgrade
#   sudo ./scripts/install-mediamtx.sh --force-config
#       also replace /usr/local/etc/mediamtx.yml if you've edited it
#
# MEDIAMTX_VERSION=v1.21.1 picks the MediaMTX release (default below).
set -euo pipefail

MEDIAMTX_VERSION="${MEDIAMTX_VERSION:-v1.21.1}"
BIN_PATH=/usr/local/bin/mediamtx
CONF_PATH=/usr/local/etc/mediamtx.yml
UNIT_PATH=/etc/systemd/system/mediamtx.service
SERVICE_USER=mediamtx

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
force_config=false

for arg in "$@"; do
  case "$arg" in
    --force-config) force_config=true ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "run this with sudo" >&2
  exit 1
fi

case "$(uname -m)" in
  aarch64) arch=arm64 ;;
  armv7l) arch=armv7 ;;
  *) echo "unsupported architecture $(uname -m); this is meant for a Raspberry Pi" >&2; exit 1 ;;
esac

tarball="mediamtx_${MEDIAMTX_VERSION}_linux_${arch}.tar.gz"
base_url="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading MediaMTX ${MEDIAMTX_VERSION} (${arch})"
curl -fsSL -o "$tmp/$tarball" "$base_url/$tarball"
curl -fsSL -o "$tmp/checksums.sha256" "$base_url/checksums.sha256"
(cd "$tmp" && grep " \*\?${tarball}\$" checksums.sha256 | sha256sum --check --status) || {
  echo "checksum mismatch for $tarball" >&2
  exit 1
}
tar -xzf "$tmp/$tarball" -C "$tmp" mediamtx

# Copy next to the target, then rename, so a running binary is swapped atomically.
install -m 0755 "$tmp/mediamtx" "${BIN_PATH}.new"
mv "${BIN_PATH}.new" "$BIN_PATH"
echo "Installed $BIN_PATH"

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin "$SERVICE_USER"
  echo "Created user $SERVICE_USER"
fi
for group in video render; do
  if getent group "$group" >/dev/null; then
    usermod -aG "$group" "$SERVICE_USER"
  fi
done

install -d -m 0755 "$(dirname "$CONF_PATH")"
if [[ ! -f "$CONF_PATH" ]] || $force_config; then
  install -m 0644 "$REPO_DIR/mediamtx/mediamtx.yml" "$CONF_PATH"
  echo "Installed $CONF_PATH"
elif ! cmp -s "$REPO_DIR/mediamtx/mediamtx.yml" "$CONF_PATH"; then
  install -m 0644 "$REPO_DIR/mediamtx/mediamtx.yml" "${CONF_PATH}.new"
  echo "Kept your edited $CONF_PATH; the repo version is at ${CONF_PATH}.new"
  echo "(rerun with --force-config to replace it)"
fi

install -m 0644 "$REPO_DIR/systemd/mediamtx.service" "$UNIT_PATH"
systemctl daemon-reload
systemctl enable mediamtx.service >/dev/null
systemctl restart mediamtx.service

echo
echo "MediaMTX is running. Useful commands:"
echo "  systemctl status mediamtx"
echo "  journalctl -u mediamtx -f"
