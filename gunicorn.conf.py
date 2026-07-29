# gunicorn.conf.py
import os

# ── Worker model ────────────────────────────────────────────────────────────
# NOT gevent. gRPC's C-core polling engine does not cooperate with gevent's
# event loop, so any Firestore call from a greenlet blocks the hub forever —
# no exception raised, so no error handler can recover. google-cloud-firestore
# is thread-safe, so threads give the same I/O concurrency without the trap.
workers      = 1
worker_class = "gthread"
threads      = 8

bind      = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
keepalive = 5
loglevel  = "info"

# Long enough for synchronous Groq parsing, short enough that a wedged worker
# is recycled and logged as WORKER TIMEOUT instead of holding the socket open.
timeout          = 120
graceful_timeout = 30

# Never preload: the master would import the app and every worker would inherit
# its gRPC channels. Each worker imports fresh after the fork.
preload_app = False


def post_fork(server, worker):
    """
    Reinitialize Firebase inside each forked worker.

    With preload_app = False there is nothing inherited to clean up, so this is
    belt-and-braces. It stays because it also guarantees the credential parsing
    happens per worker rather than depending on import order.
    """
    import json
    import firebase_admin
    from firebase_admin import credentials, firestore as fs_admin, storage

    # Drop any inherited app (no-op without preload, cheap insurance with it)
    for existing_app in list(firebase_admin._apps.values()):
        try:
            firebase_admin.delete_app(existing_app)
        except Exception:
            pass

    try:
        raw = (
            os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            or ""
        ).strip()

        if raw:
            sa_dict = json.loads(raw)
            if "private_key" in sa_dict:
                sa_dict["private_key"] = sa_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(sa_dict)
        else:
            cred = credentials.Certificate("serviceAccountKey.json")

        firebase_admin.initialize_app(cred, {
            "storageBucket": os.environ.get(
                "FIREBASE_STORAGE_BUCKET",
                "eduket.firebasestorage.app",
            )
        })

        # Log which project the credentials actually belong to. A mismatch
        # against the web app's project ID is what turns every verify_id_token
        # call into a 401.
        project_id = json.loads(raw).get("project_id") if raw else "file"
        print(f"[post_fork] Firebase ready in pid={worker.pid} project={project_id}", flush=True)

        # Keep app.py's module-level globals pointing at this worker's client.
        # Routes migrated to tier_limits.get_db() no longer depend on this.
        import app as flask_app
        flask_app.db = fs_admin.client()
        flask_app.bucket = storage.bucket()

    except Exception as e:
        print(f"[post_fork] Firebase init FAILED: {type(e).__name__}: {e}", flush=True)
        raise   # Do not serve traffic with a broken Firebase; let the worker die loudly.