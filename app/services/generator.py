# app/services/generator.py
from typing import List, Tuple
import os, re, json
from .retriever import knn, knn_state
from .ranker import final_score
from .store import get_profile
from .utils import safe_float, normalize_state
from .llm_groq import complete as groq_complete
from .llm_groq import complete_stream as groq_stream

# ---------------- Persona & knobs ----------------

STRICT_STATE = os.getenv("STRICT_STATE", "1") == "1"  # 1 = in-state only in results
BANNED_PHRASES = [r"review( the)? bills?", r"monitor(ing)? updates", r"monitor legislative updates"]

DEFAULT_STEPS = [
    "Inventory where automated tools influence decisions; document vendor, model, purpose, data, and human review.",
    "Request vendors’ latest bias-testing or impact-assessment results and prepare a one-paragraph applicant disclosure with a human-review option."
]

EMPLOYMENT_SYNONYMS = [
    "automated hiring", "automated employment decision tools", "AEDT",
    "employment screening", "recruiting algorithms", "bias audit", "audit",
]

# Policy explainer
SYS = (
     "You are an AI policy explainer for business executives. "
     "Write 5–7 crisp sentences in plain English. "
     "State what the retrieved bills mean in practice for employers: who is covered; key obligations (audits, disclosures, notices, risk assessments); penalties or private right of action; effective dates/status (e.g., pending/enacted). "
     "If the state has no bill directly covering the user’s use case, say that plainly and (only if requested) mention a nearby model bill. "
     "Do not tell the user to 'review' anything or to 'monitor updates'. Avoid legalese. "
     "End with exactly two concrete next steps written as imperative verbs. Not legal advice."
)

# Friendly general assistant (for “hi”, “how are you”, etc.)
SYS_GENERAL = (
    "You are a helpful, concise assistant. Be friendly, practical, and upbeat. "
    "Answer in 4–6 short sentences unless the user asks for code or lists."
)

# ---------------- Helpers ----------------

GREET_WORDS = {"hi", "hello", "hey", "hola", "yo", "sup", "how are you", "good morning", "good evening", "good afternoon"}
GENERAL_HINT_WORDS = {"how", "what", "why", "explain", "help", "best", "tips", "difference", "compare", "write", "fix", "error"}
LAW_HINT_WORDS = {"ai", "bill", "law", "act", "regulation", "hiring", "privacy", "biometric", "automated", "compliance", "audit"}

def _is_greeting(msg: str) -> bool:
    q = (msg or "").strip().lower()
    if not q:
        return False
    # very short messages or classic greet words
    return (len(q) <= 6 and any(q.startswith(w[: len(q)]) for w in GREET_WORDS)) or any(w in q for w in GREET_WORDS)

def _looks_general(q: str) -> bool:
    ql = (q or "").lower()
    # treat as general if it lacks law-ish hints and contains generic help words
    return (not any(w in ql for w in LAW_HINT_WORDS)) and any(w in ql for w in GENERAL_HINT_WORDS)

def _postprocess_exec_style(text: str) -> str:
    # postprocess only for policy answers
    t = (text or "").strip()
    # remove banned phrases
    for pat in BANNED_PHRASES:
        t = re.sub(pat, "", t, flags=re.I)
    # normalize whitespace
    t = re.sub(r"[ \t]+\n", "\n", t).strip()
    # enforce exactly two next steps
    steps = re.findall(r"^(?:-|\d+\.)\s*(.+)$", t, flags=re.M)
    if len(steps) != 2:
        t = re.sub(r"(?is)\n?\s*next steps:.*$", "", t).strip()
        t += "\n\nNext steps:\n- " + DEFAULT_STEPS[0] + "\n- " + DEFAULT_STEPS[1]
    # ensure disclaimer
    if "Not legal advice" not in t:
        t += "\n\nNot legal advice."
    return t

def _augment_query(message: str, state: str) -> str:
    # only augment when it looks like a policy/legal query
    msg = (message or "").strip()
    if _is_greeting(msg) or _looks_general(msg):
        return msg
    extra = " ".join(EMPLOYMENT_SYNONYMS)
    return f"{msg} in {state} {extra}".strip() if state else f"{msg} {extra}".strip()

def _context_block(hits: List[dict]) -> str:
    lines = []
    for h in hits:
        cats = (h.get("category") or "").replace(";", ", ")
        cats = f" | categories: {cats}" if cats else ""
        date = f" | date: {h.get('date')}" if h.get("date") else ""
        url = f" | url: {h.get('url')}" if h.get("url") else ""
        lines.append(
            f"- {h.get('bill_id','')} | {h.get('title','')} | state: {h.get('state','')}{date}{cats}{url}\n  snippet: {h.get('snippet','')}"
        )
    return "\n".join(lines)

def _suggest_examples(state: str, industry: str | None) -> List[str]:
    st = state or "your state"
    ind = (industry or "your industry").lower()
    return [
        f"Do we need to disclose AI use to applicants in {st}?",
        f"Are bias audits required for automated hiring tools in {st}?",
        f"What counts as an automated employment decision tool in {st}?",
        f"Any bills touching AI use for {ind} companies in {st}?",
        f"When would penalties or private lawsuits apply in {st}?",
    ]

def _greeting_reply(profile: dict) -> str:
    state = (profile.get("state") or "").strip() or "your state"
    industry = (profile.get("industry") or "").strip() or "your industry"
    # you asked to be addressed as “sir”
    examples = _suggest_examples(state, industry)
    lines = [
        "hi sir — I’m your AI policy explainer.",
        f"I can summarize AI bills and explain what they mean for employers in {state}, in plain English.",
        "ask me something specific, or try one of these:",
        f"- {examples[0]}",
        f"- {examples[1]}",
        f"- {examples[2]}",
        "or ask general work questions — I’m happy to help."
    ]
    return "\n".join(lines)

def _fallback_summary(profile: dict, query: str, hits: List[dict]) -> str:
    state = profile.get("state", "unspecified")
    first = [f"{h.get('bill_id','')} {h.get('title','')} ({h.get('state','')}, {h.get('date','')})" for h in hits]
    first = "; ".join([s.strip() for s in first if s.strip()])[:600]
    return (
        f"Here’s a concise, plain-English summary for {state}. The top matches include {first}. "
        "They focus on employer use of automated tools and typically require bias audits or disclosures. "
        "For decisions that affect hiring or screening, expect obligations like transparency to applicants and periodic risk assessments. "
        "Treat this as guidance, not legal advice."
    )

# ---------------- Main answer (non-stream) ----------------

def answer(session_id: str, message: str) -> Tuple[str, List[dict]]:
    profile = get_profile(session_id) or {}
    raw_state = (profile.get("state") or "").strip()
    user_state = normalize_state(raw_state)

    # 1) Greeting shortcut
    if _is_greeting(message):
        return _greeting_reply(profile), []

    # 2) General (non-legal) Q&A
    if _looks_general(message):
        messages = [
            {"role": "system", "content": SYS_GENERAL},
            {"role": "user", "content": message},
        ]
        out = groq_complete(messages)
        return (out or "Happy to help, sir."), []

    # 3) Policy / retrieval path
    query = _augment_query(message, user_state) if message else message

    # In-state first, then global
    in_state_hits = knn_state(query, user_state, k=20) if user_state else []
    global_hits = knn(query, k=30)

    # Merge de-duped
    def key(h): return (h.get("bill_id",""), h.get("state",""))
    seen, merged = set(), []
    for lst in (in_state_hits, global_hits):
        for h in lst:
            k = key(h)
            if k not in seen:
                seen.add(k); merged.append(h)

    # Re-rank
    rescored: List[dict] = []
    for r in merged:
        same_state = bool(user_state) and (normalize_state(r.get("state")) == user_state)
        bill_cats = {c.strip().lower() for c in (r.get("category") or "").split(";") if c.strip()}
        is_hiring = any(k in (query or "").lower() for k in ["hiring", "employment", "aedt", "screening"])
        preferred = set(profile.get("categories", [])) or ({"effect on labor/employment", "private sector use"} if is_hiring else set())
        cat_match = bool(preferred and (preferred & bill_cats))
        score = final_score(r.get("sim", 0.0), same_state, cat_match, r.get("date", ""))
        r2 = dict(r); r2["score"] = safe_float(score); rescored.append(r2)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    if user_state:
        in_state = [x for x in rescored if normalize_state(x.get("state")) == user_state]
        out_state = [x for x in rescored if normalize_state(x.get("state")) != user_state]
        top = in_state[:6] if STRICT_STATE else ((in_state[:4] + out_state[:2]) if in_state else out_state[:6])
    else:
        top = rescored[:6]

    if user_state and STRICT_STATE and not top:
        return ("This state currently has no directly relevant items in our corpus. Ask to broaden the search if you want regional model bills.", [])

    if not top:
        return ("I couldn’t find relevant items. Try adding your state or more context about the AI use (e.g., automated hiring bias audit).", [])

    # LLM explanation via Groq
    ctx = _context_block(top)
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content":
            f"Company state: {user_state or 'unspecified'}\n"
            f"User question: {message}\n\n"
            "Relevant bills (metadata + short snippets). Summarize what they mean in practice for an employer in this state.\n"
            f"{ctx}\n\n"
            "Output:\n"
            "- 5–7 sentences, plain English, executive tone.\n"
            "- If no bill directly covers the use in this state, say so plainly. Do not discuss other states unless the user asks.\n"
            "- Mention obligations (audits, disclosures, notices, risk assessments), penalties/PRoA, and effective dates/status when present.\n"
            "- Finish with EXACTLY TWO concrete next steps (imperative verbs)."
        }
    ]
    out = groq_complete(messages)
    reply = _postprocess_exec_style(out) if out else _fallback_summary(profile, message, top)
    return reply, top

# ---------------- Streaming answer ----------------

def answer_stream(session_id: str, message: str):
    """
    Streams the reply in chunks. Ends with a marker line:
    \n||SOURCES||{"sources":[...]}
    """
    profile = get_profile(session_id) or {}
    raw_state = (profile.get("state") or "").strip()
    user_state = normalize_state(raw_state)

    # 1) Greeting shortcut (hand-crafted, no model)
    if _is_greeting(message):
        yield _greeting_reply(profile)
        yield "\n||SOURCES||" + json.dumps({"sources": []})
        return

    # 2) General (non-legal) Q&A -> stream from model
    if _looks_general(message):
        messages = [
            {"role": "system", "content": SYS_GENERAL},
            {"role": "user", "content": message},
        ]
        for chunk in groq_stream(messages):
            if chunk:
                yield chunk
        yield "\n||SOURCES||" + json.dumps({"sources": []})
        return

    # 3) Policy / retrieval path
    query = _augment_query(message, user_state) if message else message
    in_state_hits = knn_state(query, user_state, k=20) if user_state else []
    global_hits = knn(query, k=30)

    def key(h): return (h.get("bill_id",""), h.get("state",""))
    seen, merged = set(), []
    for lst in (in_state_hits, global_hits):
        for h in lst:
            k = key(h)
            if k not in seen:
                seen.add(k); merged.append(h)

    rescored: List[dict] = []
    for r in merged:
        same_state = bool(user_state) and (normalize_state(r.get("state")) == user_state)
        bill_cats = {c.strip().lower() for c in (r.get("category") or "").split(";") if c.strip()}
        is_hiring = any(k in (query or "").lower() for k in ["hiring","employment","aedt","screening"])
        preferred = set(profile.get("categories", [])) or ({"effect on labor/employment","private sector use"} if is_hiring else set())
        cat_match = bool(preferred and (preferred & bill_cats))
        score = final_score(r.get("sim", 0.0), same_state, cat_match, r.get("date", ""))
        r2 = dict(r); r2["score"] = safe_float(score); rescored.append(r2)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    if user_state:
        in_state = [x for x in rescored if normalize_state(x.get("state")) == user_state]
        out_state = [x for x in rescored if normalize_state(x.get("state")) != user_state]
        top = in_state[:6] if STRICT_STATE else ((in_state[:4] + out_state[:2]) if in_state else out_state[:6])
    else:
        top = rescored[:6]

    if user_state and STRICT_STATE and not top:
        yield "This state currently has no directly relevant items in our corpus. Ask to broaden the search if you want regional model bills."
        yield "\n||SOURCES||" + json.dumps({"sources": []})
        return

    if not top:
        yield "I couldn’t find relevant items. Try adding your state or more context about the AI use (e.g., automated hiring bias audit)."
        yield "\n||SOURCES||" + json.dumps({"sources": []})
        return

    ctx = _context_block(top)
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content":
            f"Company state: {user_state or 'unspecified'}\n"
            f"User question: {message}\n\n"
            "Relevant bills (metadata + short snippets). Summarize what they mean in practice for an employer in this state.\n"
            f"{ctx}\n\n"
            "Output:\n"
            "- 5–7 sentences, plain English, executive tone.\n"
            "- If no bill directly covers the use in this state, say so plainly. Do not discuss other states unless the user asks.\n"
            "- Mention obligations (audits, disclosures, notices, risk assessments), penalties/PRoA, and effective dates/status when present.\n"
            "- Finish with EXACTLY TWO concrete next steps (imperative verbs)."
        }
    ]
    # stream raw model text
    for chunk in groq_stream(messages):
        if chunk:
            yield chunk
    # final marker with sources
    yield "\n||SOURCES||" + json.dumps({"sources": top})
