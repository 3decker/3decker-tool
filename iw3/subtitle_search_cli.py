"""python -m iw3.subtitle_search_cli -- standalone OpenSubtitles search/download tool.

Real-world use case: a user has (or is about to make) an iw3 3D conversion of a movie
and wants a matching .srt subtitle track without hunting for one by hand. This tool
searches OpenSubtitles' official REST API (https://api.opensubtitles.com/api/v1) and
downloads a plain .srt to disk. It does SEARCH + DOWNLOAD only -- it never mixes/mutes
anything into a video file itself. Feed the resulting .srt into the already-existing
`iw3.subtitle_mux_cli` exactly as if you'd found it manually (see that module for the
actual muxing step). Kept single-purpose per this project's "one small tool per
concern" convention -- do not fold this into subtitle_mux_cli.py.

ORIGINAL SOURCE vs. CONVERTED OUTPUT (read this before touching --original-source):
OpenSubtitles' `moviehash` search parameter (the "OSHash" algorithm, see
compute_moviehash() below) is only useful against the user's ORIGINAL, pre-conversion
source video file -- NOT the already-converted iw3 SBS/TB output, which is a
completely different file (different size, different bytes) and will essentially
never hash-match anything. --original-source and the file iw3 already converted are
therefore two distinct, separately-provided things: this tool takes ONLY the
original-source path (optional -- enables moviehash search when given, falls back to
--title/--imdb-id text search when omitted). It never takes or needs the converted
output path at all -- the resulting .srt is a generic subtitle for the movie, equally
muxable into any conversion of it.

CREDENTIALS (see docs/ai/AI_DECISIONS.md ADR-039 for the full design):
- Api-Key (application-level, shared "3DECKER" app key, NOT the user's personal
  secret but still not hardcoded as a literal in this file): read from
  nunif/tmp/opensubtitles_config.json's "api_key" field, or overridden per-run with
  --api-key. This file is created automatically (with an empty placeholder) the first
  time this tool runs. To get a real key: register a free account at
  https://www.opensubtitles.com -> Account -> "API Consumers" (or the direct API
  Consumer registration page linked from there), create a new API Consumer with
  whatever name you like (the research this tool was built from used "3DECKER"), copy
  the "API Key" value it gives you, and paste it as the "api_key" value in
  nunif/tmp/opensubtitles_config.json (the file this tool creates on first run).
- Login email/password (--login-email/--login-password, optional): only used to POST
  /login and obtain a JWT for the higher logged-in download quota. The password is
  NEVER written to disk. Only the resulting JWT is cached (in the same config file,
  alongside its real expiry decoded from the token's own "exp" claim -- see
  _decode_jwt_exp()) so a normal re-run within the token's real lifetime does not need
  to log in again. Anonymous mode (both flags omitted) works too, just with a smaller
  download quota.
- Download quota: this tool never hardcodes an assumed request-per-day number --
  /download's JSON response includes real "remaining"/"requests"/"reset_time" fields
  from the server for the credentials actually used, which this tool prints after
  every download so the real, current number is always what's shown.

SEARCH / DOWNLOAD SPLIT (for a future GUI results picker):
search() returns a plain list of dicts (JSON-serializable, no network capability
attached) so a future GUI can call search() once, show the user every candidate's
metadata (release name, language, download_count, ratings, hearing_impaired, hd, fps,
from_trusted, ai_translated/machine_translated, upload_date, uploader name -- ai_/
machine_translated surfaced prominently since most users want to avoid AI-junk subs),
let the user click one, and only then call download() with that one result's
file_id -- spending exactly one download-quota credit, never speculatively. run()
(the CLI entry point) is a thin wrapper: it calls search(), and in --list-only mode
just prints the results and stops; otherwise it auto-picks one result (preferring a
non-machine/ai-translated one when any exist -- see _pick_best_result()) and calls
download() itself, mirroring what a GUI would do with the user's own click.
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from os import path

from nunif.utils.home_dir import ensure_home_dir
from nunif.utils.filename import sanitize_filename


API_BASE_URL = "https://api.opensubtitles.com/api/v1"
USER_AGENT = "3DECKER-iw3-subtitle-search/1.0"

# Same config-directory convention iw3-gui.py uses for its own persisted, non-secret
# app settings (nunif/tmp/, via ensure_home_dir) -- see docs/ai/AI_DECISIONS.md
# ADR-039 for why this file, not an env var, was chosen.
CONFIG_DIR = ensure_home_dir("iw3", path.join(path.dirname(__file__), "..", "tmp"))
CONFIG_PATH = path.join(CONFIG_DIR, "opensubtitles_config.json")

_DEFAULT_CONFIG = {
    "api_key": "",
    "jwt": {"token": None, "email": None, "expires_at": None},
}

REGISTER_KEY_INSTRUCTIONS = (
    "No OpenSubtitles API key is configured.\n"
    "To get one (free): go to https://www.opensubtitles.com -> log in or create an "
    "account -> Account -> 'API Consumers' -> register a new API Consumer (any name "
    "works, e.g. '3DECKER') -> copy the 'API Key' value it gives you.\n"
    f"Then open {CONFIG_PATH} and set its \"api_key\" field to that value, e.g.:\n"
    '  {"api_key": "your-real-key-here", "jwt": {...}}\n'
    "Alternatively pass it just for this run with --api-key <your-key>."
)

# --- moviehash (OSHash) ------------------------------------------------------------
# OpenSubtitles' documented moviehash algorithm (originated in Media Player Classic,
# still used by the current REST API's `moviehash` search parameter): file size, plus
# a 64-bit checksum formed by summing 8-byte little-endian words from ONLY the first
# 64KB and last 64KB of the file (never the whole file -- this is deliberately cheap
# even on huge files). Files under 128KB (2x the chunk size) cannot be hashed this way
# -- MOVIEHASH_MIN_FILE_SIZE below is that hard floor, handled explicitly by callers
# rather than producing a meaningless/undersized result.
MOVIEHASH_CHUNK_SIZE = 65536  # 64 KiB
MOVIEHASH_MIN_FILE_SIZE = MOVIEHASH_CHUNK_SIZE * 2  # 128 KiB
_INT64_MASK = 0xFFFFFFFFFFFFFFFF


def compute_moviehash(file_path):
    """Returns (hash_hex_str, filesize) on success, or (None, filesize) if file_path
    is smaller than MOVIEHASH_MIN_FILE_SIZE and cannot be hashed this way. Raises
    OSError if file_path cannot be opened/read at all (a real I/O problem, not an
    expected "too small" case -- left to the caller to report)."""
    filesize = os.path.getsize(file_path)
    if filesize < MOVIEHASH_MIN_FILE_SIZE:
        return None, filesize

    hash64 = filesize
    with open(file_path, "rb") as f:
        for chunk_offset in (0, filesize - MOVIEHASH_CHUNK_SIZE):
            f.seek(chunk_offset)
            data = f.read(MOVIEHASH_CHUNK_SIZE)
            for i in range(0, len(data) - 7, 8):
                (word,) = struct.unpack_from("<q", data, i)
                hash64 = (hash64 + word) & _INT64_MASK
    return f"{hash64:016x}", filesize


# --- config / credentials -----------------------------------------------------------

def _load_config(config_path=CONFIG_PATH):
    if not path.exists(config_path):
        _save_config(config_path, dict(_DEFAULT_CONFIG))
        return dict(_DEFAULT_CONFIG)
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return dict(_DEFAULT_CONFIG)
    merged = dict(_DEFAULT_CONFIG)
    merged.update(data if isinstance(data, dict) else {})
    if not isinstance(merged.get("jwt"), dict):
        merged["jwt"] = dict(_DEFAULT_CONFIG["jwt"])
    return merged


def _save_config(config_path, config):
    # Atomic write (tmp-name + os.replace) per docs/ai/CODING_STANDARDS.md CS-IO-001.
    out_dir = path.dirname(path.abspath(config_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = config_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, config_path)


def resolve_api_key(cli_api_key, config):
    """CLI --api-key (this run only, never persisted) takes priority over the config
    file's value. Returns the key string, or None if neither is set."""
    if cli_api_key:
        return cli_api_key
    key = config.get("api_key")
    return key if key else None


def _decode_jwt_exp(token):
    """Best-effort extraction of the 'exp' (Unix timestamp) claim from a JWT's own
    payload -- so the local token cache respects the token's REAL server-issued
    expiry instead of hardcoding an assumed lifetime. Returns None if the token isn't
    a well-formed JWT or carries no 'exp' claim; callers must then treat the cache as
    untrustworthy (do not reuse it)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        exp = payload.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _get_cached_jwt(config, email):
    jwt = config.get("jwt") or {}
    if not jwt.get("token") or jwt.get("email") != email:
        return None
    expires_at = jwt.get("expires_at")
    if expires_at is None or time.time() >= float(expires_at) - 60:
        return None
    return jwt["token"]


def _cache_jwt(config_path, config, email, token):
    expires_at = _decode_jwt_exp(token)
    config["jwt"] = {"token": token, "email": email, "expires_at": expires_at}
    _save_config(config_path, config)
    return expires_at


# --- HTTP -----------------------------------------------------------------------

class ApiError(Exception):
    pass


def _http_call(method, url, api_key, jwt_token=None, json_body=None, timeout=30):
    """Low-level JSON request. Returns (parsed_json_body, http_status). Raises
    ApiError with a human-readable message on any transport/HTTP/JSON-parse failure
    -- callers turn that into a CLI error message, never a raw traceback."""
    data = None
    headers = {"Api-Key": api_key, "User-Agent": USER_AGENT, "Accept": "application/json"}
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read()
    except urllib.error.URLError as e:
        raise ApiError(f"network error calling {url}: {e.reason}")

    try:
        parsed = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        parsed = {}

    if status >= 400:
        message = parsed.get("message") or parsed.get("errors") or body[:500]
        raise ApiError(f"OpenSubtitles API returned HTTP {status} for {url}: {message}")

    return parsed, status


# --- login -----------------------------------------------------------------------

def login(api_key, email, password, base_url=API_BASE_URL, timeout=30):
    """POST /login. Returns (jwt_token_str, error_message_or_None). Never persists
    the password anywhere -- the caller is responsible for caching only the returned
    token (see _cache_jwt)."""
    try:
        parsed, _status = _http_call(
            "POST", f"{base_url}/login", api_key,
            json_body={"username": email, "password": password}, timeout=timeout)
    except ApiError as e:
        return None, str(e)

    token = parsed.get("token")
    if not token:
        return None, f"login response did not contain a token: {parsed}"
    return token, None


# --- search ------------------------------------------------------------------------

_RESULT_FIELDS = (
    "release", "language", "download_count", "ratings", "hearing_impaired", "hd",
    "fps", "from_trusted", "ai_translated", "machine_translated", "upload_date",
)


def _parse_search_results(payload):
    results = []
    for item in payload.get("data", []) or []:
        attrs = item.get("attributes", {}) or {}
        files = attrs.get("files") or []
        file_id = files[0].get("file_id") if files else None
        if file_id is None:
            continue  # nothing downloadable in this entry -- skip it
        entry = {field: attrs.get(field) for field in _RESULT_FIELDS}
        entry["file_id"] = file_id
        entry["subtitle_id"] = attrs.get("subtitle_id")
        uploader = attrs.get("uploader") or {}
        entry["uploader_name"] = uploader.get("name")
        results.append(entry)
    return results


def search(api_key, moviehash=None, imdb_id=None, query=None, languages="eng",
           order_by=None, jwt_token=None, base_url=API_BASE_URL, timeout=30):
    """Returns (results, error_message_or_None). results is a plain list of
    JSON-serializable dicts (see _RESULT_FIELDS + file_id/subtitle_id/uploader_name),
    already parsed out of OpenSubtitles' JSON:API-shaped response -- callable
    independently of download() so a future GUI can show a results picker."""
    if not api_key:
        return [], REGISTER_KEY_INSTRUCTIONS

    params = {"languages": languages}
    if moviehash:
        params["moviehash"] = moviehash
    if imdb_id:
        params["imdb_id"] = imdb_id
    if query:
        params["query"] = query
    if order_by:
        params["order_by"] = order_by
    if not (moviehash or imdb_id or query):
        return [], ("ERROR: no search criteria given -- provide at least one of "
                     "--original-source (for moviehash), --imdb-id, or --title.")

    url = f"{base_url}/subtitles?{urllib.parse.urlencode(params)}"
    try:
        parsed, _status = _http_call("GET", url, api_key, jwt_token=jwt_token, timeout=timeout)
    except ApiError as e:
        return [], str(e)

    return _parse_search_results(parsed), None


def _pick_best_result(results):
    """Auto-pick logic used by run() in non-list-only mode: prefer a result that is
    neither machine_translated nor ai_translated (most users want to avoid AI-junk
    subs by default -- see module docstring), falling back to the plain first result
    (the API's own ordering) if every candidate is translated. Stable -- never
    reorders within either group."""
    if not results:
        return None
    for r in results:
        if not r.get("machine_translated") and not r.get("ai_translated"):
            return r
    return results[0]


# --- download ------------------------------------------------------------------------

def download(api_key, file_id, output_dir, jwt_token=None, output_filename=None,
             base_url=API_BASE_URL, timeout=30):
    """POST /download (spends exactly one download-quota credit -- only call this
    once you actually want the file, never speculatively), then GET the returned
    temporary link and write the .srt to output_dir. Returns
    (saved_path_or_None, error_message_or_None, quota_info_dict).

    quota_info_dict surfaces the server's OWN reported "remaining"/"requests"/
    "reset_time" fields verbatim (no hardcoded assumption about the daily cap --
    see module docstring) -- empty dict if the response didn't include them."""
    if not api_key:
        return None, REGISTER_KEY_INSTRUCTIONS, {}

    try:
        parsed, _status = _http_call(
            "POST", f"{base_url}/download", api_key, jwt_token=jwt_token,
            json_body={"file_id": file_id}, timeout=timeout)
    except ApiError as e:
        return None, str(e), {}

    link = parsed.get("link")
    if not link:
        return None, f"download response did not contain a link: {parsed}", {}

    quota_info = {k: parsed[k] for k in ("remaining", "requests", "reset_time") if k in parsed}

    try:
        req = urllib.request.Request(link, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
    except urllib.error.URLError as e:
        return None, f"failed to fetch subtitle content from temporary link: {e}", quota_info

    filename = sanitize_filename(
        output_filename or parsed.get("file_name") or f"opensubtitles_{file_id}.srt")
    if not filename.lower().endswith(".srt"):
        filename += ".srt"

    os.makedirs(output_dir, exist_ok=True)
    out_path = path.join(output_dir, filename)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(content)
    os.replace(tmp_path, out_path)

    return out_path, None, quota_info


# --- CLI -----------------------------------------------------------------------

def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.subtitle_search_cli",
        description=(
            "Search OpenSubtitles' official REST API for a subtitle and download it as a "
            "plain .srt file. Search-and-download only -- does not mux the result into any "
            "video; feed the .srt into iw3.subtitle_mux_cli separately for that. "
            + REGISTER_KEY_INSTRUCTIONS),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--original-source", type=str, default=None,
                         help="Path to the ORIGINAL, pre-conversion source video file (NOT an "
                              "already-converted iw3 SBS/TB output -- that will not hash-match "
                              "anything). Optional: if given and at least 128KB, its OpenSubtitles "
                              "moviehash is computed and used as a search parameter for exact-match "
                              "results. If omitted, falls back to --title/--imdb-id text search.")
    parser.add_argument("--imdb-id", type=str, default=None,
                         help="IMDb ID to search by (e.g. tt0111161 or 111161). Optional.")
    parser.add_argument("--title", type=str, default=None,
                         help="Movie/show title to search by (fallback text query). Optional.")
    parser.add_argument("--language", type=str, default="eng",
                         help="ISO 639 language code to search for (e.g. eng, jpn, fre).")
    parser.add_argument("--order-by", type=str, default="download_count",
                         help="OpenSubtitles result ordering field (e.g. download_count, "
                              "ratings, upload_date).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Directory to save the downloaded .srt into. Default: the "
                              "directory containing --original-source if given, else the "
                              "current directory.")
    parser.add_argument("--api-key", type=str, default=None,
                         help="OpenSubtitles application API key, overriding the value in "
                              f"{CONFIG_PATH} for this run only (not persisted). " + REGISTER_KEY_INSTRUCTIONS)
    parser.add_argument("--login-email", type=str, default=None,
                         help="OpenSubtitles account email. Optional -- if given (with "
                              "--login-password), logs in for the higher registered-user "
                              "download quota; the resulting JWT is cached (not the password). "
                              "If omitted, runs anonymously with the smaller anonymous quota.")
    parser.add_argument("--login-password", type=str, default=None,
                         help="OpenSubtitles account password. Never written to disk -- only "
                              "the JWT obtained by logging in with it is cached.")
    parser.add_argument("--list-only", action="store_true",
                         help="Search and print candidate results without downloading anything "
                              "(spends zero download-quota credits). Useful for a future GUI "
                              "results picker calling search() directly instead.")
    return parser


def _format_result_line(i, r):
    flags = []
    if r.get("machine_translated"):
        flags.append("MACHINE-TRANSLATED")
    if r.get("ai_translated"):
        flags.append("AI-TRANSLATED")
    if r.get("hearing_impaired"):
        flags.append("HI")
    if r.get("from_trusted"):
        flags.append("trusted")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    return (f"  [{i}] file_id={r['file_id']} lang={r.get('language')} "
            f"downloads={r.get('download_count')} ratings={r.get('ratings')} "
            f"hd={r.get('hd')} fps={r.get('fps')} upload_date={r.get('upload_date')}"
            f"{flag_str}\n      release: {r.get('release')}\n      uploader: {r.get('uploader_name')}")


def run(args):
    config = _load_config(CONFIG_PATH)
    api_key = resolve_api_key(args.api_key, config)
    if not api_key:
        print(f"ERROR: {REGISTER_KEY_INSTRUCTIONS}", file=sys.stderr)
        return 1

    jwt_token = None
    if args.login_email and args.login_password:
        jwt_token = _get_cached_jwt(config, args.login_email)
        if jwt_token:
            print(f"[subtitle-search] reusing cached login token for {args.login_email}", file=sys.stderr)
        else:
            print(f"[subtitle-search] logging in as {args.login_email}...", file=sys.stderr)
            jwt_token, login_error = login(api_key, args.login_email, args.login_password)
            if login_error:
                print(f"ERROR: login failed: {login_error}", file=sys.stderr)
                return 1
            expires_at = _cache_jwt(CONFIG_PATH, config, args.login_email, jwt_token)
            if expires_at:
                print(f"[subtitle-search] login OK, token cached until "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expires_at))}", file=sys.stderr)
            else:
                print("[subtitle-search] login OK (token expiry unknown -- will not be reused)",
                      file=sys.stderr)
    elif args.login_email or args.login_password:
        print("ERROR: --login-email and --login-password must both be given, or both omitted "
              "(anonymous mode).", file=sys.stderr)
        return 1

    moviehash = None
    if args.original_source:
        if not path.exists(args.original_source):
            print(f"ERROR: --original-source file does not exist: {args.original_source}", file=sys.stderr)
            return 1
        moviehash, filesize = compute_moviehash(args.original_source)
        if moviehash is None:
            print(f"[subtitle-search] --original-source is only {filesize} bytes (minimum "
                  f"{MOVIEHASH_MIN_FILE_SIZE} for moviehash) -- skipping hash search, falling "
                  f"back to title/IMDb-ID text search.", file=sys.stderr)
        else:
            print(f"[subtitle-search] computed moviehash: {moviehash}", file=sys.stderr)

    results, error = search(
        api_key, moviehash=moviehash, imdb_id=args.imdb_id, query=args.title,
        languages=args.language, order_by=args.order_by, jwt_token=jwt_token)
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not results:
        print("[subtitle-search] no results found.", file=sys.stderr)
        return 1

    print(f"[subtitle-search] {len(results)} result(s):", file=sys.stderr)
    for i, r in enumerate(results):
        print(_format_result_line(i, r), file=sys.stderr)

    if args.list_only:
        return 0

    chosen = _pick_best_result(results)
    if chosen.get("machine_translated") or chosen.get("ai_translated"):
        print("[subtitle-search] WARNING: every result is machine/AI-translated -- downloading "
              "the top one anyway.", file=sys.stderr)
    print(f"[subtitle-search] downloading file_id={chosen['file_id']} ({chosen.get('release')})...",
          file=sys.stderr)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = path.dirname(path.abspath(args.original_source)) if args.original_source else os.getcwd()

    saved_path, dl_error, quota_info = download(
        api_key, chosen["file_id"], output_dir, jwt_token=jwt_token)
    if dl_error:
        print(f"ERROR: {dl_error}", file=sys.stderr)
        return 1

    print(f"[subtitle-search] done -- wrote {saved_path}", file=sys.stderr)
    if quota_info:
        print(f"[subtitle-search] download quota (server-reported): {quota_info}", file=sys.stderr)
    return 0


# --- self tests --------------------------------------------------------------------

def _self_test_moviehash():
    """Determinism/correctness of compute_moviehash() -- no network needed. Uses a
    synthetic file well over the 128KB floor plus one deliberately under it."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="iw3_subsearch_selftest_") as tmpdir:
        big_path = path.join(tmpdir, "big.bin")
        with open(big_path, "wb") as f:
            f.write(os.urandom(200_000))

        h1, size1 = compute_moviehash(big_path)
        h2, size2 = compute_moviehash(big_path)
        assert h1 is not None and len(h1) == 16, h1
        assert h1 == h2, "hash must be deterministic across repeated calls"
        assert size1 == size2 == 200_000

        # Changing content changes the hash.
        with open(big_path, "r+b") as f:
            f.seek(0)
            f.write(b"\x00")
        h3, _ = compute_moviehash(big_path)
        assert h3 != h1, "hash must change when file content changes"

        # Below the 128KB floor -> explicit None, not a bogus/partial hash.
        small_path = path.join(tmpdir, "small.bin")
        with open(small_path, "wb") as f:
            f.write(b"x" * 1000)
        h_small, size_small = compute_moviehash(small_path)
        assert h_small is None, h_small
        assert size_small == 1000

        # Exactly at the floor must succeed (boundary check).
        boundary_path = path.join(tmpdir, "boundary.bin")
        with open(boundary_path, "wb") as f:
            f.write(os.urandom(MOVIEHASH_MIN_FILE_SIZE))
        h_boundary, _ = compute_moviehash(boundary_path)
        assert h_boundary is not None and len(h_boundary) == 16

    print("_self_test_moviehash: PASS")


def _self_test_missing_api_key():
    """search()/download() must give a clear, actionable error -- not a crash or a
    silent no-op -- when no API key is configured."""
    results, error = search(None, query="Some Movie")
    assert results == []
    assert "opensubtitles.com" in error and CONFIG_PATH in error

    saved, error, quota = download("", file_id=123, output_dir=".")
    assert saved is None
    assert "opensubtitles.com" in error

    print("_self_test_missing_api_key: PASS")


def _self_test_search_parsing():
    """Mocked-HTTP test of search()'s request-building and response-parsing, using a
    realistic (trimmed) OpenSubtitles JSON:API-shaped payload -- no real network call
    (per docs/ai/TEST_MATRIX.md's synthetic/isolated preference)."""
    from unittest.mock import patch, MagicMock

    fake_payload = {
        "total_pages": 1,
        "total_count": 2,
        "data": [
            {
                "id": "1", "type": "subtitle",
                "attributes": {
                    "subtitle_id": "1001", "language": "en", "download_count": 500,
                    "ratings": 9.2, "hearing_impaired": False, "hd": True, "fps": 23.976,
                    "from_trusted": True, "ai_translated": False, "machine_translated": False,
                    "upload_date": "2020-01-01T00:00:00Z",
                    "release": "Movie.2020.1080p.BluRay.x264-GROUP",
                    "uploader": {"uploader_id": 42, "name": "someuser"},
                    "files": [{"file_id": 12345, "cd_number": 1, "file_name": "movie.srt"}],
                },
            },
            {
                "id": "2", "type": "subtitle",
                "attributes": {
                    "subtitle_id": "1002", "language": "en", "download_count": 10,
                    "ratings": 5.0, "hearing_impaired": False, "hd": False, "fps": None,
                    "from_trusted": False, "ai_translated": True, "machine_translated": False,
                    "upload_date": "2021-01-01T00:00:00Z",
                    "release": "Movie.2020.WEBRip-AI",
                    "uploader": {"uploader_id": 43, "name": "aiuser"},
                    "files": [{"file_id": 99999, "cd_number": 1, "file_name": "movie_ai.srt"}],
                },
            },
        ],
    }

    class FakeResponse:
        def __init__(self, body, status=200):
            self._body = body
            self.status = status

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        return FakeResponse(json.dumps(fake_payload).encode("utf-8"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        results, error = search("real-api-key", moviehash="abc123", languages="eng",
                                 order_by="download_count", jwt_token="fake-jwt")
    assert error is None, error
    assert len(results) == 2
    assert results[0]["file_id"] == 12345
    assert results[0]["release"] == "Movie.2020.1080p.BluRay.x264-GROUP"
    assert results[0]["machine_translated"] is False
    assert results[1]["ai_translated"] is True

    # Correct headers were sent (Api-Key, Bearer JWT, User-Agent).
    req = captured_requests[0]
    assert req.get_header("Api-key") == "real-api-key", dict(req.header_items())
    assert req.get_header("Authorization") == "Bearer fake-jwt"
    assert req.get_header("User-agent") == USER_AGENT
    assert "moviehash=abc123" in req.full_url
    assert "order_by=download_count" in req.full_url

    best = _pick_best_result(results)
    assert best["file_id"] == 12345, "must prefer the non-AI-translated result"

    print("_self_test_search_parsing: PASS")


def _self_test_download_flow():
    """Mocked-HTTP test of download()'s two-step flow (POST /download for the link,
    then plain GET for content) and its quota-info passthrough."""
    import tempfile
    from unittest.mock import patch, MagicMock

    class FakeResponse:
        def __init__(self, body, status=200):
            self._body = body
            self.status = status

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    download_response = json.dumps({
        "link": "https://dl.opensubtitles.com/fake/link.srt",
        "file_name": "Movie.2020.srt",
        "remaining": 9,
        "requests": 10,
        "reset_time": "23h59m",
    }).encode("utf-8")
    srt_content = b"1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "opensubtitles.com/fake/link.srt" in req.full_url:
            return FakeResponse(srt_content)
        return FakeResponse(download_response)

    with tempfile.TemporaryDirectory(prefix="iw3_subsearch_selftest_") as tmpdir:
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            saved_path, error, quota = download("real-api-key", file_id=12345, output_dir=tmpdir)
        assert error is None, error
        assert saved_path is not None and path.exists(saved_path)
        with open(saved_path, "rb") as f:
            assert f.read() == srt_content
        assert quota == {"remaining": 9, "requests": 10, "reset_time": "23h59m"}
        assert len(calls) == 2, "must POST /download, then a separate GET for content"

    print("_self_test_download_flow: PASS")


def _self_test_jwt_expiry_cache():
    """_decode_jwt_exp / _get_cached_jwt / _cache_jwt round trip -- no network."""
    import tempfile

    payload = {"exp": time.time() + 3600}
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    fake_jwt = f"header.{payload_b64}.signature"

    exp = _decode_jwt_exp(fake_jwt)
    assert exp is not None and abs(exp - payload["exp"]) < 1

    assert _decode_jwt_exp("not-a-jwt") is None

    with tempfile.TemporaryDirectory(prefix="iw3_subsearch_selftest_") as tmpdir:
        cfg_path = path.join(tmpdir, "cfg.json")
        config = dict(_DEFAULT_CONFIG)
        _cache_jwt(cfg_path, config, "user@example.com", fake_jwt)

        reloaded = _load_config(cfg_path)
        assert _get_cached_jwt(reloaded, "user@example.com") == fake_jwt
        assert _get_cached_jwt(reloaded, "someone-else@example.com") is None

        # An expired token must not be reused.
        expired_payload = {"exp": time.time() - 3600}
        expired_b64 = base64.urlsafe_b64encode(json.dumps(expired_payload).encode()).rstrip(b"=").decode()
        expired_jwt = f"header.{expired_b64}.signature"
        _cache_jwt(cfg_path, config, "user@example.com", expired_jwt)
        reloaded = _load_config(cfg_path)
        assert _get_cached_jwt(reloaded, "user@example.com") is None

    print("_self_test_jwt_expiry_cache: PASS")


def _run_self_tests():
    _self_test_moviehash()
    _self_test_missing_api_key()
    _self_test_search_parsing()
    _self_test_download_flow()
    _self_test_jwt_expiry_cache()
    print("All subtitle_search_cli self-tests PASSED")


def main(argv=None):
    # Special-cased ahead of the real parser (rather than added as a parser argument)
    # so it can run with no other required args -- see docs/ai/TEST_MATRIX.md's
    # "isolated (no-GPU) test pattern" convention.
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
