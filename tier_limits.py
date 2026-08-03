"""
tier_limits.py — Dynamic Seat-Based Quotas & Usage Tracking
═══════════════════════════════════════════════════════════════════════════════
Replaces hardcoded tiers with dynamic seat-based calculations (Students + Teachers).
Uses server-side Firestore aggregation (.count()) for high-performance usage reads.
"""

import logging
import os
import threading
from datetime import datetime, timezone

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

log = logging.getLogger(__name__)

FIRESTORE_TIMEOUT = 8.0

# ── Baseline Quotas Per Seat ──────────────────────────────────────────────────
DEFAULT_EXAMS_PER_STUDENT = 2  # Monthly exam upload quota per purchased student seat
DEFAULT_EXAMS_PER_TEACHER = 2  # Monthly exam upload quota per purchased teacher seat
FREE_TIER_MONTHLY_LIMIT = 4  # Free/trial tier default monthly exam upload quota

# Exams are scoped to the current calendar month; headcounts are standing total seats
MONTHLY_SCOPED = {"exams", "exam"}

# ── Lazy Per-Process Firestore Client ─────────────────────────────────────────
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


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _count(query) -> int:
    """
    Server-side count aggregation query. Bills 1 read per 1,000 documents
    instead of pulling full document payloads into memory.
    """
    try:
        result = query.count().get(timeout=FIRESTORE_TIMEOUT)
        return int(result[0][0].value)
    except AttributeError:
        # Fallback for older google-cloud-firestore SDKs lacking aggregation
        log.warning("Aggregation query unavailable, falling back to stream()")
        return sum(1 for _ in query.stream(timeout=FIRESTORE_TIMEOUT))


def _month_start_iso() -> str:
    """
    Returns the ISO-8601 string for the 1st day of the current UTC month.
    Exams store `uploadedAt` as an ISO string, enabling fast string filter queries.
    """
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()


# ── Dynamic Limit Calculation Engine ─────────────────────────────────────────

def get_school_exam_limit(school_id: str) -> int:
    """
    Calculates monthly exam upload limit based on active purchased seats
    or returns custom/overridden limit if set on the school subscription.
    """
    if not school_id:
        return FREE_TIER_MONTHLY_LIMIT

    try:
        db = get_db()
        sub_doc = db.collection("subscriptions").document(school_id).get(timeout=FIRESTORE_TIMEOUT)

        if sub_doc.exists:
            sub_data = sub_doc.to_dict() or {}

            # 1. Custom explicit limit override takes precedence if set
            if "customExamLimit" in sub_data:
                return int(sub_data["customExamLimit"])

            # 2. Seat-based dynamic calculation
            if sub_data.get("status") == "active":
                seats = sub_data.get("seats", {})
                students = int(seats.get("students", 0))
                teachers = int(seats.get("teachers", 0))

                calculated_limit = (students * DEFAULT_EXAMS_PER_STUDENT) + (teachers * DEFAULT_EXAMS_PER_TEACHER)
                return max(calculated_limit, FREE_TIER_MONTHLY_LIMIT)

        return FREE_TIER_MONTHLY_LIMIT

    except Exception as e:
        log.error("[Quota Calculation] Error calculating exam limit for school %s: %s", school_id, e)
        return FREE_TIER_MONTHLY_LIMIT


def count_school_usage(school_id: str, resource: str) -> int:
    """
    Single source of truth for counting resource usage across all endpoints.

    Supported resources:
      - 'exams' or 'exam' (Counts uploads in the current UTC calendar month)
      - 'teachers' or 'teacher' (Counts active registered teacher profiles)
      - 'students' or 'student' (Counts active registered student profiles)
    """
    if not school_id:
        return 0

    db = get_db()
    res = resource.lower().rstrip("s")

    if res == "exam":
        q = (db.collection("exams")
             .where(filter=FieldFilter("schoolId", "==", school_id))
             .where(filter=FieldFilter("uploadedAt", ">=", _month_start_iso())))
        return _count(q)

    # For 'teacher' or 'student' headcounts
    q = (db.collection("users")
         .where(filter=FieldFilter("schoolId", "==", school_id))
         .where(filter=FieldFilter("role", "==", res)))
    return _count(q)


def check_school_exam_quota(school_id: str) -> tuple[bool, int, int]:
    """
    Evaluates current month exam upload quota.
    Returns: (can_upload: bool, used: int, limit: int)
    """
    limit = get_school_exam_limit(school_id)
    used = count_school_usage(school_id, "exams")
    return (used < limit), used, limit


# ── Master Gatekeeper Evaluation ──────────────────────────────────────────────

def check_school_limit(school_id: str, limit_type: str) -> tuple[bool, str]:
    """
    Evaluates capacity for a requested resource/seat or exam upload.
    Used for pre-checks and authoritative write guardrails.
    """
    if not school_id:
        return False, "No school ID associated with request."

    db = get_db()
    res = limit_type.lower().rstrip("s")

    # 1. Fetch subscription details
    try:
        sub_doc = db.collection("subscriptions").document(school_id).get(timeout=FIRESTORE_TIMEOUT)
        sub_data = sub_doc.to_dict() if sub_doc.exists else {}
    except Exception as e:
        log.error("[Limit Check] Subscription lookup failed for school %s: %s", school_id, e)
        return False, "Unable to verify school subscription status."

    status = sub_data.get("status", "unpaid")
    seats = sub_data.get("seats", {})
    purchased_students = int(seats.get("students", 0))
    purchased_teachers = int(seats.get("teachers", 0))

    # --------------------------------------------------------------------------
    # CHECK 1: Teacher Registration Seats
    # --------------------------------------------------------------------------
    if res == "teacher":
        if status != "active" and purchased_teachers == 0:
            return False, "Active subscription required to register teachers."

        teacher_count = count_school_usage(school_id, "teachers")

        if teacher_count >= purchased_teachers:
            return False, f"Teacher seat limit reached ({teacher_count}/{purchased_teachers}). Please upgrade your seat allocation."

        return True, "Allowed"

    # --------------------------------------------------------------------------
    # CHECK 2: Student Registration Seats
    # --------------------------------------------------------------------------
    elif res == "student":
        if status != "active" and purchased_students == 0:
            return False, "Active subscription required to register students."

        student_count = count_school_usage(school_id, "students")

        if student_count >= purchased_students:
            return False, f"Student seat limit reached ({student_count}/{purchased_students}). Please upgrade your seat allocation."

        return True, "Allowed"

    # --------------------------------------------------------------------------
    # CHECK 3: Exam Monthly Generation/Upload Quota
    # --------------------------------------------------------------------------
    elif res == "exam":
        can_upload, used, limit = check_school_exam_quota(school_id)
        if not can_upload:
            return False, f"Monthly exam quota reached ({used}/{limit}). Upgrade seats to increase quota."
        return True, "Allowed"

    return True, "Allowed"