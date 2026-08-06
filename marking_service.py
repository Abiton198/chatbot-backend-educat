import os
import time
from google import genai
from google.genai import types

# Initialize client with current SDK
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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


def create_rubric_cache(ttl_minutes: int = 120):
    """
    Creates an in-memory cached context for the exam rubric.
    TTLs are refreshed or auto-expire after the session finishes.
    """
    logger_msg = f"Creating context cache (TTL: {ttl_minutes}m)..."
    print(logger_msg)

    cache = client.caches.create(
        model="gemini-3.5-flash-lite",
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


def mark_student_submission(cache_name: str, student_id: str, student_answers: str):
    """
    Evaluates a single student's script using the cached rubric.
    """
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
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            temperature=0.1,  # Low variance for consistent grading
        )
    )

    # Token usage metadata inspectable via response.usage_metadata
    usage = response.usage_metadata
    print(f"\n--- Marking Results [{student_id}] ---")
    print(f"Cached Input Tokens Used: {usage.cached_content_token_count}")
    print(f"New Input Tokens Used:    {usage.prompt_token_count - (usage.cached_content_token_count or 0)}")
    print(f"Output Tokens Generated:  {usage.candidates_token_count}")

    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE (Batch marking a class)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1: Cache the memo once before starting the grading run
    rubric_cache = create_rubric_cache(ttl_minutes=60)

    # Step 2: Batch process student submissions against the cached rubric
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
        result = mark_student_submission(
            cache_name=rubric_cache.name,
            student_id=sub["student_id"],
            student_answers=sub["answers"]
        )
        print(result)

    # Optional: Delete cache when marking session completes early
    # client.caches.delete(name=rubric_cache.name)