# gunicorn.conf.py — Eduket OS  v6.0  (Gemini, memory-conscious)
#
# Tuned for a 512 MB Render instance. The expensive things in this service are
# LibreOffice (150–250 MB per conversion) and PyMuPDF page rendering — not the
# AI calls, which are HTTP requests that spend their time waiting.

import os

# ── Worker model ────────────────────────────────────────────────────────────
# NOT gevent. gRPC's C-core polling engine does not cooperate with gevent's
# event loop, so a Firestore call from a greenlet blocks the hub forever with
# no exception raised — nothing can catch it. google-cloud-firestore is
# thread-safe, so threads give the same I/O concurrency without the trap.
#
# One worker. Two processes each load Flask, firebase-admin, gRPC, PyMuPDF and
# google-genai — roughly 200–300 MB resident before doing any work — which is
# what triggered the OOM restart. Threads inside one process share all of it.
workers      = 1
worker_class = "gthread"

# 8 -> 4. Every in-flight request can hold a PDF in memory, and extraction runs
# on top of this in its own background threads. Four is ample for a pilot:
# requests are short here, because extraction is handed off asynchronously.
threads = 4

bind      = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
keepalive = 5
loglevel  = "info"

# ── Memory guardrails ───────────────────────────────────────────────────────
# The single most effective control. PyMuPDF and LibreOffice leave fragmented
# heap behind; recycling the worker returns it to the OS. Jitter stops a
# predictable restart from landing mid-upload every time.
max_requests        = 300
max_requests_jitter = 50

# ── Timeouts ────────────────────────────────────────────────────────────────
# Extraction runs in a daemon thread and returns immediately, so uploads are
# fast. The long request is /submit, which marks each open question with a
# separate Gemini call — a 20-question paper can take 40–60s.
timeout          = 120
graceful_timeout = 30

# Never preload: the master would import the app and every worker would inherit
# its gRPC channels and its Gemini client. Each worker imports fresh after fork.
preload_app = False


def post_fork(server, worker):
    """
    Kept deliberately light.

    With preload_app = False and workers = 1 there is nothing inherited to
    clean up, and app.py calls _init_firebase() at import — which the worker
    does immediately after this hook. The heavy version of this hook (delete
    every app, re-parse credentials, re-import app.py to patch its module
    globals) duplicated that work for no benefit.

    What stays is the one thing worth knowing at boot: which Firebase project
    the credentials belong to. A mismatch against the web app's project is what
    turns every verify_id_token call into a 401, and it is invisible otherwise.
    """
    import json

    raw = (
        os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        or ""
    ).strip()

    if not raw:
        print(f"[post_fork] pid={worker.pid} — no service account in env, "
              f"falling back to serviceAccountKey.json", flush=True)
        return

    try:
        project_id = json.loads(raw).get("project_id", "unknown")
        print(f"[post_fork] pid={worker.pid} firebase_project={project_id}",
              flush=True)
    except json.JSONDecodeError:
        # Fail loudly here rather than letting app.py's import raise something
        # less obvious. A malformed credential blob is a config mistake.
        print(f"[post_fork] pid={worker.pid} FIREBASE_SERVICE_ACCOUNT_JSON "
              f"is not valid JSON", flush=True)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# IF MEMORY IS STILL TIGHT, IN ORDER OF EFFECT
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. CAP CONCURRENT EXTRACTIONS IN app.py.
#    _launch_pipeline spawns an unbounded daemon thread per upload. Ten
#    teachers uploading at once means ten threads, each holding a PDF and
#    possibly waiting on LibreOffice. Add at module level:
#
#        _EXTRACTION_SEMAPHORE = threading.Semaphore(2)
#
#    and wrap the body of run_extraction_pipeline in
#    `with _EXTRACTION_SEMAPHORE:`. Uploads queue instead of stacking.
#    This is the biggest remaining exposure and it costs nothing.
#
# 2. LOWER THE UPLOAD SIZE CAP.
#    extraction_engine.MAX_FILE_SIZE_BYTES is 50 MB. Real exam papers are under
#    5 MB; 15 MB is a generous ceiling and cuts worst-case memory by more
#    than half.
#
# 3. threads = 2.
#    Requests are short. Below this you start queueing the health check.
#
# 4. Only then upgrade the instance.
#    Watch Render's memory graph across a few real uploads first — a number
#    beats a guess, and the three changes above are free.
# ══════════════════════════════════════════════════════════════════════════════