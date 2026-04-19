"""
sentence_engine.py
Primary  : Groq free API (llama-3.1-8b-instant)
Fallback : Smart rule-based engine (works offline, no install needed)

SETUP (one time):
  pip install groq
  set GROQ_API_KEY=gsk_your_key_here   (get free key at console.groq.com)
"""

import json, re, os

# ── System prompt ─────────────────────────────────────────────────────
_SYSTEM = """You are a sentence generator for a real-time sign language recognition system.

The user signs words one at a time. You receive them as an ordered list and must produce exactly 3 natural English sentences that capture what they are communicating.

CRITICAL RULES:
- Use EVERY word. Never skip any — including: no, not, please, water, food, help, etc.
- Respect word order and negation. "no water please" = refusing water politely.
- Output ONLY a JSON array of 3 strings. No explanation. No markdown.

Examples:
["hello", "food"] -> ["Hello, I would like some food.", "Hello, can I have food please?", "Hi, I want food."]
["no", "water", "please"] -> ["No thank you, I do not want water.", "Please, no water for me.", "I am fine, no water needed."]
["thank you", "food"] -> ["Thank you for the food.", "Thank you, I enjoyed the food.", "The food was great, thank you."]
"""

# ── Backend detection ─────────────────────────────────────────────────
_backend = "rules"
_groq_client = None

_groq_key = os.environ.get("GROQ_API_KEY", "").strip()
if _groq_key:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=_groq_key)
        _backend = "groq"
        print("[SentenceEngine] Backend: GROQ (free cloud AI)")
    except ImportError:
        print("[SentenceEngine] groq package missing -> run: pip install groq")
        print("[SentenceEngine] Falling back to rule-based engine.")
else:
    print("[SentenceEngine] No GROQ_API_KEY found -> using smart rule-based engine.")
    print("[SentenceEngine] For AI sentences: get free key at console.groq.com")
    print("[SentenceEngine]   then run: set GROQ_API_KEY=gsk_...")

# ── Helpers ───────────────────────────────────────────────────────────
def _dedup(words):
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def _parse_json(raw):
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        r = json.loads(raw)
        if isinstance(r, list) and r:
            return [str(s) for s in r[:3]]
    except:
        pass
    found = re.findall(r'"([^"]{10,})"', raw)
    return found[:3] if found else None

# ── Groq call ─────────────────────────────────────────────────────────
def _groq_gen(words):
    resp = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"Words: {json.dumps(words)}"},
        ],
        max_tokens=250,
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()
    result = _parse_json(raw)
    return result if result else _rules(words)

# ── Smart rule-based fallback ─────────────────────────────────────────
# Intent taxonomy
_INTENTS = {
    "greeting":  {"hello", "hi"},
    "gratitude": {"thank you", "thanks"},
    "positive":  {"yes"},
    "negative":  {"no"},
    "request":   {"please", "help", "want", "more"},
    "hunger":    {"hungry", "eat"},
    "finish":    {"finish", "done"},
    "object":    {"food", "water", "drink"},
}
_W2I = {w: i for i, ws in _INTENTS.items() for w in ws}

def _intents(words):
    return {_W2I[w] for w in words if w in _W2I}

def _obj(words):
    for w in reversed(words):
        if w in _INTENTS["object"]: return w
    return None

def _rules(words):
    ws  = set(words)
    it  = _intents(words)
    o   = _obj(words) or "that"
    neg = "negative" in it

    # ── Exact combination rules ────────────────────────────────────
    # Gratitude + Object
    if "thank you" in ws and "food" in ws:
        return ["Thank you for the food.", "Thank you, I enjoyed the food.", "The food was great, thank you."]
    if "thank you" in ws and "water" in ws:
        return ["Thank you for the water.", "Thank you, I needed that water.", "The water was refreshing, thank you."]
    if "thank you" in ws and "help" in ws:
        return ["Thank you for your help.", "I appreciate your help, thank you.", "Thank you so much for helping me."]

    # Negation + Object (must check before positive combos)
    if "no" in ws and "water" in ws and "please" in ws:
        return ["No water for me, please.", "Please, no water, thank you.", "I am fine without water, please."]
    if "no" in ws and "food" in ws and "please" in ws:
        return ["No food for me, please.", "Please, no food, thank you.", "I am fine, no food needed please."]
    if "no" in ws and "water" in ws:
        return ["No, I do not want water.", "No water for me, thank you.", "I do not need water."]
    if "no" in ws and "food" in ws:
        return ["No, I do not want food.", "No food for me, thank you.", "I am not hungry."]

    # Greeting + Object
    if "hello" in ws and "food" in ws:
        return ["Hello, I would like some food.", "Hello, can I have food please?", "Hi, I want food."]
    if "hello" in ws and "water" in ws:
        return ["Hello, I need some water.", "Hello, can I have water please?", "Hi, I am thirsty."]
    if "hello" in ws and "help" in ws:
        return ["Hello, I need some help.", "Hello, can you help me?", "Hi, please help me."]

    # Request + Object
    if "please" in ws and "water" in ws:
        return ["Please give me water.", "Can I have water please?", "I would like some water, please."]
    if "please" in ws and "food" in ws:
        return ["Please give me food.", "Can I have food please?", "I would like some food, please."]
    if "want" in ws and "food" in ws:
        return ["I want food.", "I would like some food.", "Can I have food?"]
    if "want" in ws and "water" in ws:
        return ["I want water.", "I would like some water.", "Can I have water?"]

    # Yes + Object
    if "yes" in ws and "food" in ws:
        return ["Yes, I want food.", "Yes please, I would like food.", "Yes, I am hungry."]
    if "yes" in ws and "water" in ws:
        return ["Yes, I want water.", "Yes please, some water.", "Yes, I am thirsty."]

    # More + Object
    if "more" in ws and "food" in ws:
        return ["I want more food.", "Can I have more food please?", "More food please."]
    if "more" in ws and "water" in ws:
        return ["I want more water.", "Can I have more water please?", "More water please."]

    # Finish combos
    if "finish" in ws and "thank you" in ws:
        return ["I am done, thank you.", "Finished, thank you very much.", "That is all, thank you."]
    if "finish" in ws and "food" in ws:
        return ["I have finished my food.", "I am done eating.", "I finished the food."]

    # Hungry combos
    if "hungry" in ws and "please" in ws:
        return ["I am hungry, please give me food.", "Please, I am very hungry.", "I need food please, I am hungry."]

    # ── Intent-pair rules ─────────────────────────────────────────
    if "gratitude" in it and "object" in it:
        return [f"Thank you for the {o}.", f"Thank you, I appreciate the {o}.", f"The {o} was great, thank you."]
    if "greeting"  in it and "object" in it:
        return [f"Hello, I would like {o}.", f"Hello, can I have {o}?", f"Hi, I need {o}."]
    if "request"   in it and "object" in it:
        return [f"Please give me {o}.", f"I would like {o} please.", f"Can I have {o}?"]
    if "negative"  in it and "object" in it:
        return [f"No, I do not want {o}.", f"No {o} for me, thank you.", f"I do not need {o}."]
    if "positive"  in it and "object" in it:
        return [f"Yes, I want {o}.", f"Yes please, {o}.", f"I would like {o}."]

    # ── Single intent ─────────────────────────────────────────────
    if "gratitude" in it: return ["Thank you!", "Thank you very much.", "Thanks!"]
    if "greeting"  in it: return ["Hello! How are you?", "Hello!", "Hi there!"]
    if "positive"  in it: return ["Yes.", "Yes, please.", "Yes, I agree."]
    if "negative"  in it: return ["No.", "No, thank you.", "No, I am fine."]
    if "request"   in it: return ["Please help me.", "I need help.", "Can you help me please?"]
    if "hunger"    in it: return ["I am hungry.", "I want to eat.", "Can I have food?"]
    if "finish"    in it: return ["I am done.", "I have finished.", "That is all."]
    if "object"    in it: return [f"I want {o}.", f"Can I have {o}?", f"Please give me {o}."]

    # Fallback — still uses all words, just formatted better
    joined = " ".join(words)
    return [f"I want to say: {joined}.", f"Please help me with: {joined}.", f"I mean: {joined}."]


# ── Public API ────────────────────────────────────────────────────────
def generate_sentences(words: list) -> list:
    if not words:
        return []
    words = _dedup([str(w) for w in words])
    try:
        if _backend == "groq":
            return _groq_gen(words)
        return _rules(words)
    except Exception as e:
        print(f"[SentenceEngine] Error: {e}")
        return _rules(words)
