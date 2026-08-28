import os
import re
import time
import json
from google import genai
from google.genai import types
from groq import Groq, RateLimitError as GroqRateLimitError, \
    APIStatusError as GroqAPIStatusError, APIError as GroqAPIError

# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER SETUP — Groq primary, Gemini paid rescue
# ══════════════════════════════════════════════════════════════════════════════
# Same routing shape as extract_exams_v2.py's _generate(): Groq first, Gemini
# only when Groq is unconfigured, in a post-429 cooldown, over its rolling TPM
# budget, or actually errors out. This duplicates that script's TPM/cooldown
# logic rather than importing it — same DUPLICATION WARNING applies as the
# one already in extract_exams_v2.py's docstring: move both into a shared
# ai_routing.py that both files import, before they drift.
#
# CONFIRM BEFORE RUNNING: GROQ_MODEL_MARK below defaults to the same
# "openai/gpt-oss-120b" slug used for extraction. Verify against
# https://console.groq.com/docs/models — marking may be lighter-weight
# per call than extraction and could run acceptably on a smaller/cheaper
# Groq model if you want to tune cost further; this defaults to the
# already-confirmed extraction model rather than guessing at a second one.

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_MARK = os.getenv("GEMINI_MODEL_MARK", "gemini-3.6-flash")
GEMINI_CACHE_MODEL = os.getenv("GEMINI_CACHE_MODEL", "gemini-3.5-flash-lite")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_MARK = os.getenv("GROQ_MODEL_MARK", "groq/compound-mini")
GROQ_TPM_BUDGET = int(os.getenv("GROQ_TPM_BUDGET", "50000"))
GROQ_COOLDOWN_SECONDS = int(os.getenv("GROQ_COOLDOWN_SECONDS", "90"))

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_groq_client: Groq | None = None


def groq_client() -> Groq | None:
    global _groq_client
    if not GROQ_API_KEY:
        return None
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


_groq_usage_log: list[tuple[float, int]] = []
_groq_cooldown_until: float = 0.0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _groq_budget_ok(estimated_tokens: int) -> bool:
    now = time.time()
    global _groq_usage_log
    _groq_usage_log = [(t, n) for t, n in _groq_usage_log if now - t < 60]
    used = sum(n for _, n in _groq_usage_log)
    return (used + estimated_tokens) <= GROQ_TPM_BUDGET


def _groq_record_usage(tokens: int) -> None:
    _groq_usage_log.append((time.time(), tokens))


def _groq_in_cooldown() -> bool:
    return time.time() < _groq_cooldown_until


def _groq_start_cooldown() -> None:
    global _groq_cooldown_until
    _groq_cooldown_until = time.time() + GROQ_COOLDOWN_SECONDS
    print(f"    Groq cooldown started — routing to Gemini for the next "
          f"{GROQ_COOLDOWN_SECONDS}s")


def _parse_json_response(raw: str) -> dict:
    """Groq's json_object mode is looser than Gemini's response_schema — same
    codefence-strip defensive parse used in extract_exams_v2.py."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return json.loads(text)

# 1. Define the heavy context (Rubric / Memo / Exam Paper)
EXAM_MEMO_CONTEXT = """
EXAM MEMORANDUM & MARKING GUIDELINES
Subject: Computer Applications Technology (CAT) - Grade 12
Paper: Theory Final Revision

QUESTION 1: DATABASE VALIDATION (10 Marks)
1.1 Explain the difference between Field Size and Validation Rule.
   - Field Size: Sets maximum characters stored (e.g., Text size 20). [1 mark]
   - Validation Rule: Expression that limits input values (e.g., >0 AND <100). [1 mark]

1.2 Write an Access Validation Rule for dates after 01 January 2026.
   - Answer: >#2026/01/01# or >#2026-01-01# [1 mark]

1.3 Identify TWO properties to prevent null entries.
   - Required = Yes [1 mark]
   - Allow Zero Length = No [1 mark]

QUESTION 2: NETWORKS & SECURITY (10 Marks)
2.1 Define Firewall and explain its primary function.
   - Hardware/software filtering network traffic based on security rules. [2 marks]
2.2 Explain Two-Factor Authentication (2FA).
   - Security process requiring two distinct forms of identification before access. [2 marks]
"""


_gemini_cache = None  # created lazily — only spent on if/when Groq needs rescuing


def create_rubric_cache(ttl_minutes: int = 120):
    """
    Creates an in-memory cached context for the exam rubric.
    TTLs are refreshed or auto-expire after the session finishes.

    Only called when Groq has actually needed rescuing (see
    _ensure_gemini_cache below) — the original script created this
    unconditionally at startup, which meant paying for a Gemini cache even
    on a run where Groq alone handled every submission.
    """
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not set — Groq needed rescuing but there's no "
            "rescue provider configured. Set GEMINI_API_KEY, or investigate "
            "why Groq alone isn't handling this run (see the printed reason "
            "above each fallback)."
        )

    logger_msg = f"Creating context cache (TTL: {ttl_minutes}m)..."
    print(logger_msg)

    cache = client.caches.create(
        model=GEMINI_CACHE_MODEL,
        config=types.CreateCachedContentConfig(
            contents=[EXAM_MEMO_CONTEXT],
            # Inform the model about the role of this cached block
            system_instruction="You are an expert automated CAT exam evaluator. Mark student answers strictly against this memorandum.",
            ttl=f"{ttl_minutes * 60}s",
        )
    )
    print(f"✓ Cache Created successfully! Name: {cache.name}")
    print(f"✓ Expiration: {cache.expire_time}")
    return cache


def _ensure_gemini_cache(ttl_minutes: int = 120):
    """Singleton accessor — creates the Gemini cache on first rescue, reuses
    it for every rescue after that within this process."""
    global _gemini_cache
    if _gemini_cache is None:
        _gemini_cache = create_rubric_cache(ttl_minutes=ttl_minutes)
    return _gemini_cache


def _mark_gemini_cached(student_id: str, student_answers: str):
    """
    Evaluates a single student's script using the cached rubric.
    Rescue path — only reached when Groq is unconfigured, in cooldown, over
    its TPM budget, or actually errors. Lazily creates the Gemini cache on
    first use via _ensure_gemini_cache() rather than paying for it upfront.
    """
    cache = _ensure_gemini_cache()

    prompt = f"""
    Evaluate the following student submission for Student ID: {student_id}.

    STUDENT ANSWERS:
    {student_answers}

    Provide output in structured JSON format with:
    - total_score
    - breakdown: list of objects (question, mark_awarded, max_mark, feedback)
    """

    # Pass the cache name directly into GenerateContentConfig
    response = client.models.generate_content(
        model=GEMINI_MODEL_MARK,
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=cache.name,
            response_mime_type="application/json",
            temperature=0.1,  # Low variance for consistent grading
        )
    )

    # Token usage metadata inspectable via response.usage_metadata
    usage = response.usage_metadata
    print(f"    [gemini] cached={usage.cached_content_token_count} "
          f"new_in={usage.prompt_token_count - (usage.cached_content_token_count or 0)} "
          f"out={usage.candidates_token_count}")

    return response.text


def _mark_groq(student_id: str, student_answers: str):
    """
    Primary path. Groq has no equivalent to Gemini's context caching, so the
    rubric is sent in full on every call — fine at EXAM_MEMO_CONTEXT's
    current size (a couple of short questions). If the real memo grows to a
    full multi-page marking guideline, re-sending it on every submission
    stops being "cheap and fast" and starts being wasteful — that's exactly
    the kind of heavy-context case Gemini's caching exists for, so route
    large rubrics to the Gemini path deliberately rather than force Groq to
    eat it. (Not implemented here since the current rubric doesn't warrant
    it — flagging so this isn't a silent trap later.)
    """
    gc = groq_client()
    if gc is None:
        raise RuntimeError("GROQ_API_KEY not set")

    prompt = f"""{EXAM_MEMO_CONTEXT}

You are an expert automated CAT exam evaluator. Mark the student submission
below strictly against the memorandum above.

Evaluate the following student submission for Student ID: {student_id}.

STUDENT ANSWERS:
{student_answers}

Respond with ONLY a single JSON object (no markdown fences, no commentary)
with:
- total_score
- breakdown: list of objects (question, mark_awarded, max_mark, feedback)
"""

    resp = gc.chat.completions.create(
        model=GROQ_MODEL_MARK,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    usage = getattr(resp, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None) or _estimate_tokens(prompt)
    _groq_record_usage(total_tokens)
    if usage:
        print(f"    [groq] in={usage.prompt_tokens} out={usage.completion_tokens}")

    parsed = _parse_json_response(resp.choices[0].message.content)
    return json.dumps(parsed)  # keep the same str-return shape as the Gemini path


def mark_student_submission(student_id: str, student_answers: str):
    """
    Groq-primary, Gemini-rescue dispatcher. Same routing decisions as
    extract_exams_v2.py's _generate():
      1. No GROQ_API_KEY -> Gemini, unconditionally.
      2. Active cooldown or estimated tokens would exceed the rolling TPM
         budget -> Gemini, without attempting Groq.
      3. Otherwise try Groq. RateLimitError or a 413/429-style
         APIStatusError starts a cooldown and falls back to Gemini for this
         call; any other Groq error falls back without starting a cooldown
         (not evidence Groq is out of budget).
    """
    if not GROQ_API_KEY:
        return _mark_gemini_cached(student_id, student_answers)

    estimated = _estimate_tokens(EXAM_MEMO_CONTEXT) + _estimate_tokens(student_answers)

    if _groq_in_cooldown():
        print(f"    [{student_id}] Groq in cooldown — routing to Gemini")
        return _mark_gemini_cached(student_id, student_answers)

    if not _groq_budget_ok(estimated):
        print(f"    [{student_id}] would exceed {GROQ_TPM_BUDGET} TPM budget — routing to Gemini")
        return _mark_gemini_cached(student_id, student_answers)

    try:
        return _mark_groq(student_id, student_answers)
    except GroqRateLimitError as e:
        print(f"    [{student_id}] Groq rate limit hit: {e}")
        _groq_start_cooldown()
        return _mark_gemini_cached(student_id, student_answers)
    except GroqAPIStatusError as e:
        if e.status_code in (413, 429):
            print(f"    [{student_id}] Groq status {e.status_code}: {e}")
            _groq_start_cooldown()
        else:
            print(f"    [{student_id}] Groq error {e.status_code}, falling back this call only: {e}")
        return _mark_gemini_cached(student_id, student_answers)
    except (GroqAPIError, json.JSONDecodeError) as e:
        print(f"    [{student_id}] Groq call failed ({type(e).__name__}: {e}), falling back to Gemini")
        return _mark_gemini_cached(student_id, student_answers)


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE (Batch marking a class)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    groq_status = (f"{GROQ_MODEL_MARK} (TPM budget {GROQ_TPM_BUDGET})"
                   if GROQ_API_KEY else "not configured — every call goes to Gemini")
    print("="*64)
    print(f"Primary:  Groq — {groq_status}")
    print(f"Rescue:   Gemini — {GEMINI_MODEL_MARK} (cache created only if/when needed)")
    print("="*64)

    # Batch process student submissions. No upfront Gemini cache creation —
    # it's created lazily inside _mark_gemini_cached() the first time Groq
    # actually needs rescuing, so a run where Groq alone handles everything
    # never touches the paid Gemini path at all.
    submissions = [
        {
            "student_id": "STU_101",
            "answers": "1.1 Field size sets length. Validation rule limits value. 1.2 >#2026/01/01# 2.1 Firewall blocks hackers."
        },
        {
            "student_id": "STU_102",
            "answers": "1.1 Both do the same thing. 1.2 >=2026 2.1 Hardware filtering network traffic based on rules."
        }
    ]

    for sub in submissions:
        print(f"\n--- Marking {sub['student_id']} ---")
        result = mark_student_submission(
            student_id=sub["student_id"],
            student_answers=sub["answers"]
        )
        print(result)

    # Optional: Delete the Gemini cache if this run created one
    # if _gemini_cache is not None:
    #     client.caches.delete(name=_gemini_cache.name)
    3