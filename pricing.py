import math
from datetime import datetime, timezone

# Reference Rates (in ZAR)
RATES = {
    "student_monthly": 32.0,
    "teacher_monthly": 105.0,
    "base_platform_fee_monthly": 500.0,  # Minimum base subscription
    "extra_exam_pack_price": 150.0,  # e.g., R150 per 10 additional AI exam extractions
    "extra_exam_pack_size": 10,
}

DISCOUNTS = {
    "monthly": 0.0,
    "quarterly": 0.05,  # 5%
    "annual": 0.10,  # 10%
}

CYCLE_MONTHS = {
    "monthly": 1,
    "quarterly": 3,
    "annual": 12,
}


def calculate_subscription_quote(
        students: int,
        teachers: int,
        cycle: str = "annual",
        additional_exam_packs: int = 0
) -> dict:
    """
    Calculates the total cost for a subscription billing cycle.
    """
    if cycle not in CYCLE_MONTHS:
        raise ValueError(f"Invalid billing cycle: {cycle}")

    months = CYCLE_MONTHS[cycle]
    discount = DISCOUNTS[cycle]

    # Monthly un-discounted costs
    monthly_students = students * RATES["student_monthly"]
    monthly_teachers = teachers * RATES["teacher_monthly"]
    monthly_base = RATES["base_platform_fee_monthly"]

    monthly_subtotal = monthly_base + monthly_students + monthly_teachers

    # Multiply across billing duration
    period_subtotal = monthly_subtotal * months

    # Apply cycle discount (Quarterly 5%, Annual 10%)
    discount_amount = period_subtotal * discount
    period_total = period_subtotal - discount_amount

    # Optional AI Exam Add-ons
    addon_cost = additional_exam_packs * RATES["extra_exam_pack_price"]
    total_due = period_total + addon_cost

    return {
        "cycle": cycle,
        "months": months,
        "student_seats": students,
        "teacher_seats": teachers,
        "monthly_equivalent": round(total_due / months, 2),
        "subtotal_before_discount": round(period_subtotal, 2),
        "discount_applied": round(discount_amount, 2),
        "addon_exam_packs_cost": round(addon_cost, 2),
        "total_due_now": round(total_due, 2),
    }


def calculate_prorated_user_addon(
        current_seats: int,
        additional_seats: int,
        seat_type: str,  # "student" or "teacher"
        cycle: str,
        days_remaining: int,
        total_days_in_period: int = 365
) -> dict:
    """
    Calculates the prorated amount when a school adds users mid-subscription.
    """
    rate_per_month = (
        RATES["student_monthly"] if seat_type == "student" else RATES["teacher_monthly"]
    )

    months = CYCLE_MONTHS.get(cycle, 12)
    discount = DISCOUNTS.get(cycle, 0.10)

    # Full period cost for the new seats with the cycle discount applied
    full_period_cost = (additional_seats * rate_per_month * months) * (1 - discount)

    # Prorate based on remaining days in the active period
    prorated_amount = (days_remaining / total_days_in_period) * full_period_cost

    return {
        "seat_type": seat_type,
        "new_seats_added": additional_seats,
        "total_seats_after_update": current_seats + additional_seats,
        "days_remaining": days_remaining,
        "prorated_amount_due": round(max(prorated_amount, 0.0), 2),
    }