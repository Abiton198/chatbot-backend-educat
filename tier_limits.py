import logging
import os
import threading
from datetime import datetime, timezone

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

log = logging.getLogger(__name__)

FIRESTORE_TIMEOUT = 8.0

# ── Lazy per-process client ─────────────────────────────────────────────────
# A module-level `db = firestore.client()` is inherited across gunicorn's fork.
# The child gets a gRPC channel it doesn't own: .stream() blocks forever and
# raises nothing, so no exception handler can catch it. Build on first use.

_db = None
_db_lock = threading.Lock()


def get_db():
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = firestore.client()
                log.info("Firestore client created in pid %s", os.getpid())
    return _db


# ── Tier table ──────────────────────────────────────────────────────────────

LIMITS = {
    "free":     {"exams":   4, "teachers":  2, "students":   30},
    "silver":   {"exams":  15, "teachers":  5, "students":  150},
    "gold":     {"exams":  30, "teachers": 10, "students":  300},
    "platinum": {"exams":  80, "teachers": 25, "students":  800},
    "diamond":  {"exams": 150, "teachers": 50, "students": 2000},
}

# Exams are a monthly quota; people are a standing headcount.
MONTHLY_SCOPED = {"exams"}


def _count(query) -> int:
    """
    Server-side count. Aggregation queries bill one read per 1000 documents
    and transfer no document bodies — the previous len(list(.stream())) pulled
    every matching doc, so a 2000-student school paid 2000 reads per check.
    """
    try:
        result = query.count().get(timeout=FIRESTORE_TIMEOUT)
        return int(result[0][0].value)
    except AttributeError:
        # Older google-cloud-firestore without aggregation support.
        log.warning("aggregation unavailable, falling back to stream()")
        return sum(1 for _ in query.stream(timeout=FIRESTORE_TIMEOUT))


def _month_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def count_school_usage(school_id: str, resource: str) -> int:
    """
    Single source of truth for usage counts. Import this from /exams/upload as
    well — two independent counting implementations will drift, and the one in
    the pre-check will start refusing what the upload endpoint allows.
    """
    db = get_db()

    if resource == "exams":
        q = db.collection("exams").where(filter=FieldFilter("schoolId", "==", school_id))
        if "exams" in MONTHLY_SCOPED:
            q = q.where(filter=FieldFilter("createdAt", ">=", _month_start()))
        return _count(q)

    role_target = resource[:-1]   # 'teachers' → 'teacher'
    q = (db.collection("users")
           .where(filter=FieldFilter("schoolId", "==", school_id))
           .where(filter=FieldFilter("role", "==", role_target)))
    return _count(q)


def check_school_limit(school_id: str, limit_type: str) -> tuple:
    """
    Returns (is_allowed, error_message).

    Fails open on infrastructure errors — the authoritative enforcement lives
    at the write endpoint, so a degraded Firestore should not stop teaching.
    """
    if not school_id:
        return False, "School ID is required."

    # ── 1. Normalize the key first — cheap, and avoids a wasted read ────────
    key = (limit_type or "").lower().strip()
    if key in ("teacher", "student", "exam"):
        key = f"{key}s"
    if key not in ("exams", "teachers", "students"):
        log.warning("unknown limit_type %r — failing open", limit_type)
        return True, ""

    # ── 2. School document ─────────────────────────────────────────────────
    try:
        school_doc = get_db().collection("schools").document(school_id).get(timeout=FIRESTORE_TIMEOUT)
    except Exception as exc:
        log.warning("school fetch failed for %s: %s: %s", school_id, type(exc).__name__, exc)
        return True, ""

    if not school_doc.exists:
        return False, "School not found. Please ask your principal to register the school first."

    school_data = school_doc.to_dict() or {}

    # Accept the field under any of the names the schema has used. A silent
    # miss here reads as 'free' and throttles a paying school to 4 exams.
    raw_tier = (school_data.get("tier")
                or school_data.get("tierId")
                or school_data.get("subscriptionTier")
                or "")
    tier_id = str(raw_tier).lower().strip()

    if tier_id not in LIMITS:
        log.warning("school %s has unrecognised tier %r — applying free limits",
                    school_id, raw_tier)
        tier_id = "free"

    max_allowed = LIMITS[tier_id].get(key, 0)

    # ── 3. Count ───────────────────────────────────────────────────────────
    try:
        current = count_school_usage(school_id, key)
    except Exception as exc:
        log.warning("count failed for %s/%s: %s: %s", school_id, key, type(exc).__name__, exc)
        return True, ""

    window = " this month" if key in MONTHLY_SCOPED else ""
    log.info("[limit] %s | %s | %s: %s/%s%s", school_id, tier_id, key, current, max_allowed, window)

    if current >= max_allowed:
        return False, (
            f"Your school has reached the {tier_id.capitalize()} plan limit of "
            f"{max_allowed} {key}{window}. Ask your principal to upgrade the plan "
            f"to add more {key}."
        )

    return True, ""