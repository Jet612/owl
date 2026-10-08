#!/usr/bin/env bash
# Check an owl API the way the website's viewers reach it.
#
# Usage:
#   OWL_API_URL=https://<pi-name>.<tailnet>.ts.net OWL_API_SECRET=... ./scripts/smoke-test.sh
#
# Run it from a machine that is NOT on the tailnet: that is how viewers reach the
# Pi through Tailscale Funnel. On the Pi itself you can test the API directly:
#   set -a; . /etc/owl/owl.env; set +a
#   OWL_API_URL=http://127.0.0.1:8080 ./scripts/smoke-test.sh
#
# Optional:
#   OWL_EXPECT_CAMERA=1   count "camera offline" as a failure (use when the camera should be up)
#   OWL_SITE_ORIGIN=...   the website's origin for the CORS checks (default https://owl.example;
#                         set it if OWL_CORS_ORIGINS on the Pi is not "*")
#
# Nothing here changes anything on the Pi. Needs bash, curl and openssl.
set -uo pipefail

API="${OWL_API_URL:-}"
SECRET="${OWL_API_SECRET:-}"
ORIGIN="${OWL_SITE_ORIGIN:-https://owl.example}"
if [[ -z $API || -z $SECRET ]]; then
  echo "set OWL_API_URL and OWL_API_SECRET (see the top of this script)" >&2
  exit 2
fi
for tool in curl openssl; do
  command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 2; }
done
API="${API%/}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
# Keep the secret out of the process list by passing it to curl from a file.
printf 'Authorization: Bearer %s\n' "$SECRET" > "$work/auth"
chmod 600 "$work/auth"

# A media token, minted the same way as the website's mintOwlToken.
mint() { # mint <seconds from now>
  local exp sig
  exp=$(( $(date +%s) + $1 ))
  sig=$(printf 'owl-media:%s' "$exp" | openssl dgst -sha256 -hmac "$SECRET" -binary \
        | openssl base64 -A | tr '+/' '-_' | tr -d '=')
  echo "$exp.$sig"
}
TOKEN="$(mint 3600)"
EXPIRED="$(mint -60)"
TAMPERED="${TOKEN%%.*}.$(printf 'A%.0s' {1..43})"
WRONG_SECRET="$(SECRET="not-the-secret-$SECRET" mint 3600)"

pass=0
fail=0
STATUS=""
BODY=""
HEADERS=""

# request <curl args>: sets STATUS, BODY (first 4 KB) and HEADERS from the response.
request() {
  STATUS="$(curl -s -m 20 -o "$work/body" -D "$work/headers" -w '%{http_code}' "$@")" || STATUS=000
  BODY="$(head -c 4000 "$work/body" 2>/dev/null | tr -d '\0')"
  HEADERS="$(tr -d '\r' < "$work/headers" 2>/dev/null)"
}
ok()  { pass=$((pass + 1)); printf '  PASS  %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  FAIL  %s\n' "$1"; [[ -n ${2:-} ]] && printf '        %s\n' "$2"; return 0; }
note() { printf '  note  %s\n' "$1"; }

status_is() { # status_is <label> <status>
  if [[ $STATUS == "$2" ]]; then ok "$1"; else bad "$1" "expected HTTP $2, got $STATUS: ${BODY:0:140}"; fi
}
body_has() { # body_has <label> <regex>
  if grep -Eq -- "$2" <<<"$BODY"; then ok "$1"; else bad "$1" "body does not match $2: ${BODY:0:140}"; fi
}
header_has() { # header_has <label> <regex, matched at the start of a header line, case-insensitive>
  if grep -Eiq -- "^$2" <<<"$HEADERS"; then ok "$1"; else bad "$1" "no header matching $2"; fi
}
refused() { # refused <label>: HTTP 401 with the JSON error
  status_is "$1" 401
  body_has "$1 (as JSON)" '"error": ?"unauthorized"'
}

echo "Testing $API"
echo
echo "Health and credentials"
request "$API/api/health"
status_is "health needs no credentials" 200
body_has "health says ok" '"ok": ?true'
request "$API/api/status"
refused "status without credentials is refused"
request -H "@$work/auth" "$API/api/status"
status_is "status with the secret" 200
body_has "status has the camera fields" '"camera_online": ?(true|false)'
for field in streaming recording identifying classifier current_clip last_detection retention_days \
             clip_count disk_free_bytes disk_total_bytes server_time; do
  grep -q "\"$field\"" <<<"$BODY" || bad "status has $field"
done
STATUS_BODY="$BODY"
if [[ ${OWL_EXPECT_CAMERA:-} == 1 ]]; then
  grep -Eq '"camera_online": ?true' <<<"$STATUS_BODY" && ok "camera is online" || bad "camera is online" "status says it is not"
else
  grep -Eq '"camera_online": ?true' <<<"$STATUS_BODY" || note "camera is offline right now (set OWL_EXPECT_CAMERA=1 to make that a failure)"
fi
request "$API/api/status?t=$TOKEN"
status_is "a media token opens status" 200
request "$API/api/status?t=$EXPIRED"
refused "an expired token is refused"
request "$API/api/status?t=$TAMPERED"
refused "a changed signature is refused"
request "$API/api/status?t=$WRONG_SECRET"
refused "a token from the wrong secret is refused"
request "$API/api/status?t=1791500000.%C3%A9"
refused "a malformed token is refused, not an error"
request -H "Authorization: Bearer wrong" "$API/api/status"
refused "a wrong secret is refused"

echo
echo "Clips"
request "$API/api/clips?limit=2&t=$TOKEN"
status_is "clip list" 200
body_has "clip list has clips and a cursor" '"clips": ?\[.*"next_before"'
CLIPS_BODY="$BODY"
request "$API/api/clips?limit=abc&t=$TOKEN"
status_is "a bad limit is a 400" 400
body_has "a bad limit is JSON" '"error"'
request "$API/api/labels?t=$TOKEN"
status_is "labels" 200
body_has "labels list" '"labels": ?\['
request "$API/api/clips/20200101T000000Z?t=$TOKEN"
status_is "an unknown clip is a 404" 404
body_has "an unknown clip is JSON" '"error": ?"not found"'
request -X DELETE -H "@$work/auth" "$API/api/clips/20200101T000000Z"
status_is "deleting an unknown clip is a 404" 404

CLIP_ID="$(grep -Eo '"id": ?"[0-9]{8}T[0-9]{6}Z(-[0-9]+)?"' <<<"$CLIPS_BODY" | head -1 | grep -Eo '[0-9]{8}T[0-9]{6}Z(-[0-9]+)?')"
if [[ -n $CLIP_ID ]]; then
  request -H 'Range: bytes=0-1023' -H "Origin: $ORIGIN" "$API/api/clips/$CLIP_ID/video?t=$TOKEN"
  status_is "video answers a range request with 206" 206
  header_has "video has Content-Range" 'content-range: bytes 0-'
  header_has "video says it accepts ranges" 'accept-ranges: bytes'
  header_has "video is video/mp4" 'content-type: video/mp4'
  header_has "video is readable from the website (CORS)" 'access-control-allow-origin:'
  request -H 'Range: bytes=0-0' "$API/api/clips/$CLIP_ID/download?t=$TOKEN"
  header_has "download is an attachment" 'content-disposition: attachment'
  header_has "download is an .mp4 file" 'content-disposition:.*\.mp4"'
  request "$API/api/clips/$CLIP_ID/thumb?t=$TOKEN"
  status_is "thumbnail" 200
  header_has "thumbnail is a JPEG" 'content-type: image/jpeg'
  request "$API/api/clips/$CLIP_ID?t=$TOKEN"
  status_is "one clip" 200
  body_has "the clip has its fields" '"label_categories"'
else
  note "no clips stored yet, so video, download and thumbnail were not checked"
fi

echo
echo "Live feed"
request -H "Origin: $ORIGIN" "$API/live/$TOKEN/index.m3u8"
case $STATUS in
  200)
    body_has "the live playlist is HLS" '#EXTM3U'
    header_has "the playlist is not cached" 'cache-control:.*no-(store|cache)'
    header_has "the playlist is readable from the website (CORS)" 'access-control-allow-origin:'
    ;;
  502)
    body_has "no stream is a JSON 502" '"error"'
    header_has "the 502 is readable from the website (CORS)" 'access-control-allow-origin:'
    if [[ ${OWL_EXPECT_CAMERA:-} == 1 ]]; then bad "the live feed is up" "got 502: ${BODY:0:140}"; else note "the camera or stream is down right now, so the playlist was not checked"; fi
    ;;
  *) bad "the live route answers 200 or 502" "got $STATUS: ${BODY:0:140}" ;;
esac
request "$API/live/$EXPIRED/index.m3u8"
refused "an expired token is refused on the live route"
request "$API/live/$EXPIRED/main.m3u8"
refused "...and for every file under it"

echo
echo "Browser access"
request -X OPTIONS -H "Origin: $ORIGIN" -H 'Access-Control-Request-Method: GET' \
  -H 'Access-Control-Request-Headers: authorization, range' "$API/api/clips"
status_is "CORS preflight" 204
header_has "preflight allows the website's origin" 'access-control-allow-origin:'

echo
echo "Nothing but the API is exposed"
request "$API/"
refused "the root answers nothing without credentials"
request "$API/owl/index.m3u8"
refused "the camera's HLS path is not reachable directly"
request "$API/v3/paths/list"
refused "the MediaMTX control API is not reachable"
if [[ $API == https://* ]]; then
  host="${API#https://}"
  host="${host%%/*}"
  for port in 8443 10000; do
    code="$(curl -s -m 6 -o /dev/null -w '%{http_code}' "https://$host:$port/" 2>/dev/null || true)"
    if [[ ${code:-000} == 000 ]]; then ok "Funnel is not exposing port $port"; else bad "Funnel is not exposing port $port" "got HTTP $code"; fi
  done
fi

echo
printf '%d passed, %d failed\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
