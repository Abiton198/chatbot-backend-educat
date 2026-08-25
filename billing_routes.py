"""
billing_routes.py — EduCAT Billing & Subscription API v4.1 (School & Parent Subscriptions)
═══════════════════════════════════════════════════════════════════════════════
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

# Import dynamic quote calculator engine
from pricing import calculate_subscription_quote

logger     = logging.getLogger(__name__)
billing_bp = Blueprint("billing", __name__)


# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PAYFAST_MERCHANT_ID  = os.getenv("PAYFAST_MERCHANT_ID",  "").strip()
PAYFAST_MERCHANT_KEY = os.getenv("PAYFAST_MERCHANT_KEY", "").strip()
PAYFAST_PASSPHRASE   = os.getenv("PAYFAST_PASSPHRASE",   "").strip()

if not all([PAYFAST_MERCHANT_ID, PAYFAST_MERCHANT_KEY]):
    logger.warning(
        "PayFast credentials missing in environment. "
        "Set PAYFAST_MERCHANT_ID and PAYFAST_MERCHANT_KEY for live transactions."
    )

FRONTEND_BASE_URL = os.environ.get(
    "FRONTEND_BASE_URL", "https://eduket.tech"
).rstrip("/")

BACKEND_BASE_URL = os.environ.get(
    "BACKEND_BASE_URL", "https://chatbot-backend-educat.onrender.com"
).rstrip("/")

# PayFast known IP ranges for verification
PAYFAST_IPS = {
    "197.97.145.144", "197.97.145.145", "197.97.145.146", "197.97.145.147",
    "197.97.145.148", "197.97.145.149", "197.97.145.150", "197.97.145.151",
    "41.74.179.194",  "41.74.179.195",  "41.74.179.196",  "41.74.179.197",
    "197.97.144.128",
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

def _audit(action: str, actor: str, target: str, details: dict = None):
    try:
        if details is None:
            details = {}
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
# ROUTE: Price Quote (Supports 'school' and 'parent')
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/quote", methods=["POST", "OPTIONS"])
def billing_quote():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        data                  = request.get_json() or {}
        plan_type             = str(data.get("type") or data.get("plan") or "school").lower()
        students              = int(data.get("students", 0))
        teachers              = int(data.get("teachers", 0))
        billing_cycle         = str(data.get("billingCycle", "monthly")).lower()
        additional_exam_packs = int(data.get("additionalExamPacks", 0))

        if students < 0 or teachers < 0:
            return jsonify({"error": "Student and teacher seats must be non-negative"}), 400
        if additional_exam_packs < 0:
            return jsonify({"error": "Additional exam packs must be non-negative"}), 400

        quote = calculate_subscription_quote(
            students=students,
            teachers=teachers,
            cycle=billing_cycle,
            additional_exam_packs=additional_exam_packs,
            plan_type=plan_type
        )

        return jsonify(quote), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not calculate price quote."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Initiate Payment (Supports 'school' and 'parent')
# ══════════════════════════════════════════════════════════════════════════════

@billing_bp.route("/api/billing/initiate", methods=["POST", "OPTIONS"])
def billing_initiate():
    if request.method == "OPTIONS":
        return "", 204
    try:
        uid, err = _verify_token(request)
        if err:
            return err

        data                  = request.get_json() or {}
        plan_type             = str(data.get("type") or data.get("plan") or "school").lower()
        is_parent             = plan_type in ["parent", "parent_access"]
        students              = int(data.get("students", 0))
        teachers              = int(data.get("teachers", 0))
        billing_cycle         = str(data.get("billingCycle", "monthly")).lower()
        additional_exam_packs = int(data.get("additionalExamPacks", 0))

        school_id = None
        if not is_parent:
            school_id = _get_school_id_for_uid(uid)
            if not school_id:
                return jsonify({"error": "No school associated with this account"}), 400

        quote = calculate_subscription_quote(
            students=students,
            teachers=teachers,
            cycle=billing_cycle,
            additional_exam_packs=additional_exam_packs,
            plan_type=plan_type
        )

        if not is_parent and quote.get("is_free_baseline") and additional_exam_packs == 0:
            return jsonify({"error": "Requested allocation matches free baseline. No checkout required."}), 400

        target_prefix = "PARENT" if is_parent else f"SCHOOL_{school_id[:8].upper()}"
        payment_id = f"EDUCAT_{target_prefix}_{uuid.uuid4().hex[:8].upper()}"

        if is_parent:
            item_name = f"Parent Portal Access ({billing_cycle.capitalize()})"
        else:
            item_name = f"EduCAT Subscription ({students} Students, {teachers} Teachers)"
            if additional_exam_packs > 0:
                item_name += f" + {additional_exam_packs} Exam Pack(s)"

        payment_data = {
            "merchant_id":      PAYFAST_MERCHANT_ID,
            "merchant_key":     PAYFAST_MERCHANT_KEY,
            "return_url":       f"{FRONTEND_BASE_URL}/payment/success",
            "cancel_url":       f"{FRONTEND_BASE_URL}/payment/cancel",
            "notify_url":       f"{BACKEND_BASE_URL}/api/payfast/itn",
            "m_payment_id":     payment_id,
            "amount":           f"{quote['total_due_now']:.2f}",
            "item_name":        item_name,
            "item_description": f"{billing_cycle.capitalize()} Plan ({quote['months']} month duration)",
            "custom_str1":      "parent" if is_parent else school_id,
            "custom_str2":      uid if is_parent else str(students),
            "custom_str3":      billing_cycle if is_parent else str(teachers),
            "custom_str4":      "" if is_parent else billing_cycle,
            "custom_str5":      "" if is_parent else str(additional_exam_packs),
        }

        payment_data["signature"] = _generate_payfast_signature(payment_data)

        # Record pending transaction in Firestore
        _db().collection("paymentTransactions").document(payment_id).set({
            "targetType":           "parent" if is_parent else "school",
            "schoolId":             school_id,
            "uid":                  uid,
            "students":             students,
            "teachers":             teachers,
            "billingCycle":         billing_cycle,
            "additionalExamPacks":  additional_exam_packs,
            "expectedAmount":       quote["total_due_now"],
            "status":               "pending",
            "createdAt":            fs_admin.SERVER_TIMESTAMP,
        })

        _audit("payment_initiated", uid, uid if is_parent else school_id, {
            "targetType": "parent" if is_parent else "school",
            "billingCycle": billing_cycle,
            "amount": quote["total_due_now"],
            "payment_id": payment_id,
        })

        return jsonify({
            "paymentId":   payment_id,
            "paymentData": payment_data,
            "quote":       quote
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not initiate payment."}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: PayFast ITN Webhook (Handles School & Parent Provisioning)
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

        paid_amount     = float(itn_data.get("amount_gross", 0))
        expected_amount = float(tx.get("expectedAmount", 0))

        if abs(paid_amount - expected_amount) > max(0.5, expected_amount * 0.01):
            tx_ref.set({"status": "amount_mismatch", "paidAmount": paid_amount}, merge=True)
            return "", 200

        target_type   = tx.get("targetType", "school")
        uid           = tx.get("uid")
        billing_cycle = str(tx.get("billingCycle", "monthly")).lower()

        # Calculate billing period end date
        cycle_months_map = {"monthly": 1, "quarterly": 3, "yearly": 12, "annual": 12}
        months_to_add = cycle_months_map.get(billing_cycle, 1)

        now        = datetime.now(timezone.utc)
        period_end = now + timedelta(days=months_to_add * 30)

        batch = db.batch()

        # Branch Provisioning for Parent vs School
        if target_type == "parent":
            parent_sub_ref = db.collection("parentSubscriptions").document(uid)
            batch.set(parent_sub_ref, {
                "uid": uid,
                "status": "subscribed",
                "billingCycle": billing_cycle,
                "currentPeriodStart": now.isoformat(),
                "currentPeriodEnd": period_end.isoformat(),
                "pfPaymentId": pf_payment_id,
                "updatedAt": fs_admin.SERVER_TIMESTAMP
            }, merge=True)

            logger.info("[ITN] ✓ Parent %s provisioned with %s plan", uid, billing_cycle)
        else:
            school_id = tx.get("schoolId", "")
            students = int(tx.get("students", 0))
            teachers = int(tx.get("teachers", 0))
            additional_exam_packs = int(tx.get("additionalExamPacks", 0))

            sub_ref = db.collection("subscriptions").document(school_id)
            batch.set(sub_ref, {
                "schoolId": school_id,
                "status": "active",
                "billingCycle": billing_cycle,
                "seats": {
                    "students": students,
                    "teachers": teachers
                },
                "additionalExamPacks": fs_admin.Increment(additional_exam_packs) if additional_exam_packs else additional_exam_packs,
                "currentPeriodStart": now.isoformat(),
                "currentPeriodEnd": period_end.isoformat(),
                "pfPaymentId": pf_payment_id,
                "updatedAt": fs_admin.SERVER_TIMESTAMP
            }, merge=True)

            logger.info("[ITN] ✓ School %s provisioned with %d students / %d teachers", school_id, students, teachers)

        batch.set(tx_ref, {
            "status":      "complete",
            "paidAmount":  paid_amount,
            "pfPaymentId": pf_payment_id,
            "completedAt": fs_admin.SERVER_TIMESTAMP,
        }, merge=True)

        batch.commit()

        _audit("subscription_activated", "payfast_itn", uid if target_type == "parent" else tx.get("schoolId"), {
            "targetType": target_type,
            "amount": paid_amount
        })

        return "", 200

    except Exception:
        traceback.print_exc()
        return "", 500