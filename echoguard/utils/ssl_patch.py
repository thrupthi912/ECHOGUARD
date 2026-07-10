"""
echoguard.utils.ssl_patch
~~~~~~~~~~~~~~~~~~~~~~~~~
Disables SSL certificate verification globally for environments where a
corporate proxy intercepts HTTPS traffic and causes SSLCertVerificationError.

Import this module once at the top of main.py or any entry point:

    import echoguard.utils.ssl_patch  # noqa: F401

It patches ssl, requests, urllib3, and huggingface_hub so every network
call made by any library in the process skips certificate checks.
"""

import os
import ssl
import warnings

# ── 1. Environment variables ─────────────────────────────────────────────────
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# ── 2. Python ssl module ──────────────────────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
except Exception:
    pass

try:
    _orig_create_default_context = ssl.create_default_context

    def _patched_ctx(*args, **kwargs):
        ctx = _orig_create_default_context(*args, **kwargs)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ssl.create_default_context = _patched_ctx
except Exception:
    pass

# ── 3. urllib3 ────────────────────────────────────────────────────────────────
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ── 4. requests — patch Session so every session created later skips verify ──
try:
    import requests
    from requests import Session as _Session

    _orig_request = _Session.request

    def _unverified_request(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, method, url, **kwargs)

    _Session.request = _unverified_request  # type: ignore

    # Also patch the module-level convenience functions
    _orig_get = requests.get

    def _get(url, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_get(url, **kwargs)

    requests.get = _get  # type: ignore
except Exception:
    pass

# ── 5. huggingface_hub — set verify=False on its internal session ─────────────
try:
    import huggingface_hub.file_download as _hf_dl

    if hasattr(_hf_dl, "_get_session"):
        _orig_get_session = _hf_dl._get_session

        def _patched_get_session(*a, **kw):
            sess = _orig_get_session(*a, **kw)
            sess.verify = False
            return sess

        _hf_dl._get_session = _patched_get_session  # type: ignore
except Exception:
    pass

try:
    from huggingface_hub.utils._http import _default_backend_factory as _hf_backend  # noqa
except Exception:
    pass

# Suppress any remaining InsecureRequestWarning from anywhere
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
