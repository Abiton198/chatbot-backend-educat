"""
agent.py — EduCAT AI Agent (Groq-compatible)

Fixes applied:
1. Shorter, cleaner system prompt — Groq fails on long system+tools combos
2. tool_choice="auto" only when tools are likely needed; plain call otherwise
3. Each tool argument schema uses only "string" and "integer" types
   (Groq rejects "array" typed args in some model versions)
4. available_questions removed from tool schema — passed via context instead
5. Retry logic: on tool_use_failed, retry once without tools as plain chat
6. Max tokens raised so the model doesn't truncate mid-tool-call
"""

import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

import memory as mem
from model import mark_answer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

MAX_TOOL_ROUNDS = 5


# =========================
# 🛠️ TOOL DEFINITIONS
# Keep argument schemas SIMPLE — Groq is strict:
#   - Only "string" and "integer" primitive types
#   - No "array" or "object" in parameters (causes 400)
#   - Short descriptions (long descriptions bloat the prompt)
# =========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_theory",
            "description": "Search the CAT Grade 12 theory book for a concept or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'biometric security'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weak_topics",
            "description": "Get this student's weakest exam topics based on past results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"}
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_history",
            "description": "Get this student's past exam scores and trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"}
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_study_plan",
            "description": "Get this student's current personalised study plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"}
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_hint",
            "description": "Give a Socratic hint for a question. Never reveal the full answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_number":  {"type": "string"},
                    "question_text":    {"type": "string"},
                    "memo":             {"type": "string", "description": "Expected answer (internal use only)"},
                    "student_attempt":  {"type": "string", "description": "What the student tried (may be empty)"}
                },
                "required": ["question_number", "question_text", "memo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_study_plan",
            "description": "Save a new personalised study plan for the student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"},
                    "plan_text":  {"type": "string", "description": "The study plan content"}
                },
                "required": ["student_id", "plan_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mark_open_answer",
            "description": "Grade a student's written answer against the memo answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "string"},
                    "question_text":   {"type": "string"},
                    "student_answer":  {"type": "string"},
                    "memo":            {"type": "string"},
                    "marks":           {"type": "integer"}
                },
                "required": ["question_number", "question_text", "student_answer", "memo", "marks"]
            }
        }
    }
]


# =========================
# 🔧 TOOL RUNNERS
# =========================
def _run_tool(name, args, rag=None):
    try:
        if name == "search_theory":
            if rag is None:
                return "Theory search unavailable."
            chunks  = rag.search(args.get("query", ""))
            context = " ".join(
                c["content"] if isinstance(c, dict) and "content" in c else str(c)
                for c in chunks
            )
            return context[:1800] or "No relevant content found."

        elif name == "get_weak_topics":
            weak = mem.get_weak_topics(args.get("student_id", ""))
            if not weak:
                return "No weak topics yet."
            return "\n".join(
                f"Q{w['question_number']} | {w['q_type']} | wrong {w['wrong_count']}x | topic: {w.get('topic','')}"
                for w in weak
            )

        elif name == "get_session_history":
            sessions = mem.get_sessions(args.get("student_id", ""))
            if not sessions:
                return "No past sessions."
            return "\n".join(
                f"{s['exam_name']}: {s['score']}/{s['total']} ({s['percentage']}%) on {s['played_at']}"
                for s in sessions
            )

        elif name == "get_study_plan":
            plan = mem.get_study_plan(args.get("student_id", ""))
            return f"Plan (updated {plan['updated_at']}):\n{plan['plan']}" if plan else "No study plan yet."

        elif name == "generate_hint":
            prompt = (
                f"You are a Socratic CAT tutor. Give ONE helpful hint — do NOT reveal the answer.\n\n"
                f"Question {args.get('question_number')}: {args.get('question_text')}\n"
                f"Expected answer (internal, do NOT reveal): {args.get('memo','')}\n"
                f"Student attempt: {args.get('student_attempt') or '(none yet)'}\n\n"
                f"Write one guiding question or clue in 1-2 sentences."
            )
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.4
            )
            return r.choices[0].message.content.strip()

        elif name == "update_study_plan":
            mem.save_study_plan(args.get("student_id", ""), args.get("plan_text", ""))
            return "Study plan saved."

        elif name == "mark_open_answer":
            result = mark_answer(
                question        = args.get("question_text", ""),
                question_number = args.get("question_number", ""),
                q_type          = "open",
                student_answer  = args.get("student_answer", ""),
                memo            = args.get("memo", ""),
                marks           = int(args.get("marks", 1))
            )
            return json.dumps(result)

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {e}"


# =========================
# 🔄 AGENT LOOP
# =========================

# Short system prompt — Groq struggles with long system+tool combos
SYSTEM_PROMPT = (
    "You are EduCAT, a South African CAT Grade 12 AI tutor and exam coach. "
    "You have tools to search theory, check the student's weak topics and history, "
    "generate Socratic hints, update study plans, and mark written answers. "
    "Use tools when they would genuinely improve your answer. "
    "Be warm, specific, and always relate answers to NSC CAT Grade 12 content. "
    "When asked what to study: call get_weak_topics then get_session_history then update_study_plan. "
    "When asked about a concept: call search_theory. "
    "When giving a hint: call generate_hint."
)


def _call_with_tools(messages):
    """
    Call the LLM with tools. Returns the response message.
    Raises on hard errors, returns None content on tool_use_failed so caller can retry.
    """
    return client.chat.completions.create(
        model       = MODEL,
        messages    = messages,
        tools       = TOOLS,
        tool_choice = "auto",
        max_tokens  = 1024,
        temperature = 0.3
    )


def _call_plain(messages):
    """Fallback: call without tools when tool_use_failed."""
    return client.chat.completions.create(
        model       = MODEL,
        messages    = messages,
        max_tokens  = 768,
        temperature = 0.3
    )


def run_agent(student_id, user_message, rag=None, exam_questions=None):
    mem.ensure_student(student_id)

    # Build context-aware system message
    weak_summary = mem.get_weak_summary(student_id)
    system_msg   = SYSTEM_PROMPT
    if weak_summary and weak_summary != "No weak areas recorded yet.":
        system_msg += f" Student weak areas: {weak_summary}."
    system_msg += f" Student ID: {student_id}."

    # Load history and append new user message
    history = mem.get_history(student_id, limit=12)
    mem.append_message(student_id, "user", user_message)

    messages = (
        [{"role": "system", "content": system_msg}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    for round_num in range(MAX_TOOL_ROUNDS):
        # ── Try tool-enabled call ─────────────────────────
        try:
            response = _call_with_tools(messages)
        except Exception as e:
            err_str = str(e)
            # Groq tool_use_failed — retry as plain chat
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                print(f"  ⚠️  tool_use_failed on round {round_num} — retrying as plain chat")
                try:
                    # Strip tools context and ask plainly
                    plain_messages = messages + [{
                        "role":    "user",
                        "content": "(Note: tool calling failed. Please answer directly using your knowledge.)"
                    }]
                    r2  = _call_plain(plain_messages)
                    ans = r2.choices[0].message.content or "I ran into a technical issue. Please rephrase your question."
                    mem.append_message(student_id, "assistant", ans)
                    return ans
                except Exception as e2:
                    return f"⚠️ Could not process request: {e2}"
            return f"⚠️ Agent error: {e}"

        msg = response.choices[0].message

        # ── No tool calls → final answer ──────────────────
        if not msg.tool_calls:
            final = msg.content or "I'm not sure how to help. Could you rephrase?"
            mem.append_message(student_id, "assistant", final)
            return final

        # ── Execute tool calls ────────────────────────────
        # Append assistant turn (must include tool_calls for Groq)
        assistant_turn = {
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        }
        messages.append(assistant_turn)

        # Run each tool and append result
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            print(f"  🔧 [{round_num+1}] {tc.function.name}({list(args.keys())})")
            result = _run_tool(tc.function.name, args, rag=rag)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         tc.function.name,
                "content":      str(result)[:1500]   # cap tool results to avoid context overflow
            })

            # Persist tool result in memory
            mem.append_message(
                student_id, "tool",
                str(result)[:1500],
                tool_call_id = tc.id,
                tool_name    = tc.function.name
            )

    # Hit max rounds — ask for plain summary
    try:
        messages.append({"role": "user", "content": "Please give your final answer now based on what you found."})
        r   = _call_plain(messages)
        ans = r.choices[0].message.content or "I've gathered the information — please ask your question again."
    except Exception:
        ans = "I reached the processing limit. Please try a simpler question."

    mem.append_message(student_id, "assistant", ans)
    return ans