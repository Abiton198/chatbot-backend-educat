"""
billing_routes.py — Eduket OS Billing & Subscription API v4.0 (Per-Seat Pricing)
═══════════════════════════════════════════════════════════════════════════════
Registered in app.py:
    from billing_routes import billing_bp
    app.register_blueprint(billing_bp)

Routes
──────
    POST /api/billing/quote       Dynamic per-seat price quote (students, teachers, cycle)
    POST /api/billing/initiate    Create pending transaction + PayFast form data
    POST /api/payfast/itn         PayFast ITN webhook (updates purchased seats on payment)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import hashlib
import logging
import traceback
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import requests as http_requests
from flask import Blueprint, request, jsonify

import firebase_admin
from firebase_admin import firestore as fs_admin, auth as fb_auth

logger     = logging.getLogger(__name__)
billing_bp = Blueprint("billing", __name__)


# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PAYFAST_MERCHANT_ID  = os.getenv("PAYFAST_MERCHANT_ID",  "").strip()
PAYFAST_MERCHANT_KEY = os.getenv("PAYFAST_MERCHANT_KEY", "").strip()
PAYFAST_PASSPHRASE   = os.getenv("PAYFAST_PASSPHRASE",   "").strip()

if not all([PAYFAST_MERCHANT_ID, PAYFAST_MERCHANT_KEY]):
    raise RuntimeError(
        "Missing PayFast credentials. Set PAYFAST_MERCHANT_ID and "
        "PAYFAST_MERCHANT_KEY in your environment variables."
    )

FRONTEND_BASE_URL = os.environ.get(
    "FRONTEND_BASE_URL", "https://eduket.tech"
).rstrip("/")

BACKEND_BASE_URL = os.environ.get(
    "BACKEND_BASE_URL", "https://chatbot-backend-educat.onrender.com"
).rstrip("/")

# PayFast known IP ranges
PAYFAST_IPS = {
    "197.97.145.144", "197.97.145.145", "197.97.145.146", "197.97.145.147",
    "197.97.145.148", "197.97.145.149", "197.97.145.150", "197.97.145.151",
    "41.74.179.194",  "41.74.179.195",  "41.74.179.196",  "41.74.179.197",
    "197.97.144.128",
}


# ══════════════════════════════════════════════════════════════════════════════
# PER-SEAT RATES & CYCLE CONFIG (in ZAR)
# ══════════════════════════════════════════════════════════════════════════════

RATES = {
    "student_monthly": 72.0,
    "teacher_monthly": 105.0,
    "base_platform_fee_monthly": 500.0,
}

CYCLE_CONFIG = {
    "monthly":   {"months": 1,  "discount": 0.00},  # Standard rate
    "quarterly": {"months": 3,  "discount": 0.05},  # 5% discount
    "annual":    {"months": 12, "discount": 0.10},  # 10% discount
}


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE & AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    return fs_admin.client()

def _verify_token(req) -> tuple:
    header = req.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing or malformed Authorization header"}), 401)
    try:
        decoded = fb_auth.verify_id_token(header.split("Bearer ", 1)[1].strip())
        return decoded["uid"], None
    except Exception as e:
        logger.warning("[Billing Auth] Token verification failed: %s", e)
        return None, (jsonify({"error": "Invalid or expired token"}), 401)

def _get_school_id_for_uid(uid: str):
    try:
        doc = _db().collection("users").document(uid).get()
        return doc.to_dict().get("schoolId") if doc.exists else None
    except Exception as e:
        logger.error("[Billing] schoolId lookup failed for %s: %s", uid, e)
        return None

def _audit(action: str, actor: str, target: str, details: dict = {}):
    try:
        _db().collection("auditLog").add({
            "action":    action,
            "actorUid":  actor,
            "target":    target,
            "details":   details,
            "timestamp": fs_admin.SERVER_TIMESTAMP,
            "ip": request.headers.get(
                "X-Forwarded-For", request.remote_addr or "unknown"
            ).split(",")[0].strip(),
        })
    except Exception as e:
        logger.error("[Audit] Billing log failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# PRICING CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_per_seat_price(students: int, teachers: int, cycle: str) -> dict:
    """
    Computes total subscription price for students & teachers with cycle discounts.
    """
    config   = CYCLE_CONFIG.get(cycle, CYCLE_CONFIG["annual"])
    months   = config["months"]
    discount = config["discount"]

    monthly_students = students * RATES["student_monthly"]
    monthly_teachers = teachers * RATES["teacher_monthly"]
    monthly_base     = RATES["base_platform_fee_monthly"]

    monthly_subtotal = monthly_base + monthly_students + monthly_teachers
    period_subtotal  = monthly_subtotal * months

    discount_amount = period_subtotal * discount
    total_due       = period_subtotal - discount_amount

    return {
        "students": students,
        "teachers": teachers,
        "billingCycle": cycle,
        "months": months,
        "discountPercent": int(discount * 100),
        "monthlyEquivalent": round(total_due / months, 2),
        "subtotalBeforeDiscount": round(period_subtotal, 2),
        "discountAmount": round(discount_amount, 2),
        "totalDueZar": round(total_due, 2)
    }


# ══════════════════════════════════════════════════════════════════════════════
# PAYFAST SIGNATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _generate_payfast_signature(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if k == "signature" or v is None:
            continue
        str_val = str(v).strip()
        if str_val == "":
            continue
        parts.append(f"{k}={quote_plus(str_val)}")

    param_string = "&".join(parts)
    if PAYFAST_PASSPHRASE and PAYFAST_PASSPHRASE.strip():
        param_string += f"&passphrase={quote_plus(PAYFAST_PASSPHRASE.strip())}"

    return hashlib.md5(param_string.encode("utf-8")).hexdigest()

def _verify_payfast_signature(data: dict) -> bool:
    received = data.get("signature", "")
    check    = {k: v for k, v in data.items() if k != "signature"}
    return received == _generate_payfast_signature(check)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Dynamic Price Quote
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/quote", methods=["POST", "OPTIONS"])
def billing_quote():
    """Return price quote based on requested student and teacher seats."""
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        data          = request.get_json() or {}
        students      = int(data.get("students", 0))
        teachers      = int(data.get("teachers", 0))
        billing_cycle = data.get("billingCycle", "annual").lower()

        if students < 0 or teachers < 0:
            return jsonify({"error": "Student and teacher seats must be non-negative"}), 400
        if billing_cycle not in CYCLE_CONFIG:
            return jsonify({"error": "Invalid billingCycle"}), 400

        quote = _calculate_per_seat_price(students, teachers, billing_cycle)
        return jsonify(quote), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not calculate price quote."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Initiate Per-Seat Payment
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/initiate", methods=["POST", "OPTIONS"])
def billing_initiate():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        data          = request.get_json() or {}
        students      = int(data.get("students", 0))
        teachers      = int(data.get("teachers", 0))
        billing_cycle = data.get("billingCycle", "annual").lower()

        if students <= 0 or teachers <= 0:
            return jsonify({"error": "Must purchase at least 1 student and 1 teacher seat."}), 400
        if billing_cycle not in CYCLE_CONFIG:
            return jsonify({"error": "Invalid billingCycle"}), 400

        school_id = _get_school_id_for_uid(uid)
        if not school_id:
            return jsonify({"error": "No school associated with this account"}), 400

        quote = _calculate_per_seat_price(students, teachers, billing_cycle)
        payment_id = f"EDUKET_{school_id[:8].upper()}_{uuid.uuid4().hex[:8].upper()}"

        payment_data = {
            "merchant_id":      PAYFAST_MERCHANT_ID,
            "merchant_key":     PAYFAST_MERCHANT_KEY,
            "return_url":       f"{FRONTEND_BASE_URL}/payment/success",
            "cancel_url":       f"{FRONTEND_BASE_URL}/payment/cancel",
            "notify_url":       f"{BACKEND_BASE_URL}/api/payfast/itn",
            "m_payment_id":     payment_id,
            "amount":           f"{quote['totalDueZar']:.2f}",
            "item_name":        f"Eduket Subscription ({students} Students, {teachers} Teachers)",
            "item_description": f"{billing_cycle.capitalize()} Plan ({quote['months']} months)",
            "custom_str1":      school_id,
            "custom_str2":      str(students),
            "custom_str3":      str(teachers),
            "custom_str4":      billing_cycle,
        }

        payment_data["signature"] = _generate_payfast_signature(payment_data)

        # Record pending transaction
        _db().collection("paymentTransactions").document(payment_id).set({
            "schoolId":       school_id,
            "uid":            uid,
            "students":       students,
            "teachers":       teachers,
            "billingCycle":   billing_cycle,
            "expectedAmount": quote["totalDueZar"],
            "status":         "pending",
            "createdAt":      fs_admin.SERVER_TIMESTAMP,
        })

        _audit("payment_initiated", uid, school_id, {
            "students": students, "teachers": teachers,
            "amount": quote["totalDueZar"], "payment_id": payment_id,
        })

        return jsonify({
            "paymentId":   payment_id,
            "paymentData": payment_data,
            "quote":       quote
        }), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not initiate payment."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: PayFast ITN Webhook (Provisions Seat Allocations)
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/payfast/itn", methods=["POST"])
def payfast_itn():
    try:
        itn_data = request.form.to_dict(flat=True)

        sender_ip = request.headers.get(
            "X-Forwarded-For", request.remote_addr or ""
        ).split(",")[0].strip()

        if sender_ip and sender_ip not in PAYFAST_IPS:
            logger.warning("[ITN] Request from unrecognised IP: %s", sender_ip)

        if not _verify_payfast_signature(itn_data):
            logger.warning("[ITN] Signature mismatch from IP: %s", sender_ip)
            return "", 400

        if itn_data.get("merchant_id") != PAYFAST_MERCHANT_ID:
            return "", 400

        if itn_data.get("payment_status") != "COMPLETE":
            return "", 200

        payment_id    = itn_data.get("m_payment_id", "")
        pf_payment_id = itn_data.get("pf_payment_id", "")

        if not payment_id:
            return "", 400

        db = _db()
        tx_ref  = db.collection("paymentTransactions").document(payment_id)
        tx_snap = tx_ref.get()

        if not tx_snap.exists:
            return "", 400

        tx = tx_snap.to_dict()

        if tx.get("status") == "complete":
            return "", 200  # Already processed

        school_id     = tx.get("schoolId", "")
        students      = tx.get("students", 0)
        teachers      = tx.get("teachers", 0)
        billing_cycle = tx.get("billingCycle", "annual")

        paid_amount     = float(itn_data.get("amount_gross", 0))
        expected_amount = float(tx.get("expectedAmount", 0))

        if abs(paid_amount - expected_amount) > max(0.5, expected_amount * 0.01):
            tx_ref.set({"status": "amount_mismatch", "paidAmount": paid_amount}, merge=True)
            return "", 200

        # Success - Calculate billing end date
        months_to_add = CYCLE_CONFIG.get(billing_cycle, {}).get("months", 12)
        now           = datetime.now(timezone.utc)
        period_end    = now + timedelta(days=months_to_add * 30)

        # Update school subscription details
        batch = db.batch()

        sub_ref = db.collection("subscriptions").document(school_id)
        batch.set(sub_ref, {
            "schoolId":          school_id,
            "status":            "active",
            "billingCycle":      billing_cycle,
            "seats": {
                "students": students,
                "teachers": teachers
            },
            "currentPeriodStart": now.isoformat(),
            "currentPeriodEnd":   period_end.isoformat(),
            "pfPaymentId":        pf_payment_id,
            "updatedAt":          fs_admin.SERVER_TIMESTAMP
        }, merge=True)

        batch.set(tx_ref, {
            "status":      "complete",
            "paidAmount":  paid_amount,
            "pfPaymentId": pf_payment_id,
            "completedAt": fs_admin.SERVER_TIMESTAMP,
        }, merge=True)

        batch.commit()

        _audit("seats_upgraded", "payfast_itn", school_id, {
            "students": students, "teachers": teachers, "amount": paid_amount
        })

        logger.info("[ITN] ✓ School %s provisioned with %d students / %d teachers", school_id, students, teachers)
        return "", 200

    except Exception as e:
        traceback.print_exc()
        return "", 500