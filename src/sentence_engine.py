"""
sentence_engine.py
Priority-based sentence generation module.

Matching priority (first match wins):
  1. Exact multi-word combination rules
  2. Intent-pair rules
  3. Single-intent rules
  4. Fallback
"""

# ── Intent taxonomy ───────────────────────────────────────────────────────────
INTENT_MAP = {
    "greeting":  {"hello", "hi"},
    "gratitude": {"thank you", "thanks"},
    "positive":  {"yes"},
    "negative":  {"no"},
    "request":   {"please", "help", "want", "more"},
    "hunger":    {"hungry", "eat"},
    "finish":    {"finish", "done"},
    "object":    {"food", "water", "drink"},
}

WORD_TO_INTENT = {
    word: intent
    for intent, words in INTENT_MAP.items()
    for word in words
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _deduplicate(words):
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _intents(words):
    return {WORD_TO_INTENT[w] for w in words if w in WORD_TO_INTENT}


def _obj(words):
    """Return last object-class word, or None."""
    for w in reversed(words):
        if w in INTENT_MAP["object"]:
            return w
    return None


def _fmt(suggestions):
    seen, out = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:3]


# ── Priority 1: Exact multi-word combination rules ────────────────────────────
def _exact_rules(word_set):
    # Gratitude + Object
    if "thank you" in word_set and "food" in word_set:
        return [
            "Thank you for the food.",
            "Thank you, I enjoyed the food.",
            "Thank you, the food was great.",
        ]
    if "thank you" in word_set and "water" in word_set:
        return [
            "Thank you for the water.",
            "Thank you, I needed that water.",
            "Thank you, the water was refreshing.",
        ]
    if "thank you" in word_set and "help" in word_set:
        return [
            "Thank you for your help.",
            "Thank you, I appreciate your help.",
            "Thank you so much for helping me.",
        ]
    if "thank you" in word_set and "drink" in word_set:
        return [
            "Thank you for the drink.",
            "Thank you, I enjoyed that drink.",
            "Thank you, that drink was refreshing.",
        ]

    # Greeting + Object
    if "hello" in word_set and "food" in word_set:
        return [
            "Hello, I had food.",
            "Hello, do you want food?",
            "Hello, I want food.",
        ]
    if "hello" in word_set and "water" in word_set:
        return [
            "Hello, I need water.",
            "Hello, can I have some water?",
            "Hello, I am thirsty.",
        ]
    if "hello" in word_set and "help" in word_set:
        return [
            "Hello, I need help.",
            "Hello, can you help me?",
            "Hello, please help me.",
        ]

    # Request + Object
    if "please" in word_set and "water" in word_set:
        return [
            "Please give me water.",
            "Please, I want water.",
            "Can I have water, please?",
        ]
    if "please" in word_set and "food" in word_set:
        return [
            "Please give me food.",
            "Please, I want food.",
            "Can I have food, please?",
        ]
    if "please" in word_set and "drink" in word_set:
        return [
            "Please give me something to drink.",
            "Please, I want a drink.",
            "Can I have a drink, please?",
        ]
    if "want" in word_set and "food" in word_set:
        return [
            "I want food.",
            "I would like some food.",
            "Can I have food?",
        ]
    if "want" in word_set and "water" in word_set:
        return [
            "I want water.",
            "I would like some water.",
            "Can I have water?",
        ]
    if "more" in word_set and "food" in word_set:
        return [
            "I want more food.",
            "Can I have more food?",
            "Please give me more food.",
        ]
    if "more" in word_set and "water" in word_set:
        return [
            "I want more water.",
            "Can I have more water?",
            "Please give me more water.",
        ]

    # Positive + Object
    if "yes" in word_set and "food" in word_set:
        return [
            "Yes, I want food.",
            "Yes, please give me food.",
            "Yes, I would like some food.",
        ]
    if "yes" in word_set and "water" in word_set:
        return [
            "Yes, I want water.",
            "Yes, please give me water.",
            "Yes, I am thirsty.",
        ]

    # Negative + Object
    if "no" in word_set and "food" in word_set:
        return [
            "No, I do not want food.",
            "No, I am not hungry.",
            "No food, thank you.",
        ]
    if "no" in word_set and "water" in word_set:
        return [
            "No, I do not want water.",
            "No, I am not thirsty.",
            "No water, thank you.",
        ]

    # Hunger combos
    if "hungry" in word_set and "please" in word_set:
        return [
            "I am hungry, please give me food.",
            "Please, I am very hungry.",
            "I need food, please.",
        ]
    if "hungry" in word_set and "more" in word_set:
        return [
            "I am still hungry, I want more.",
            "I want more food.",
            "Can I have more, please?",
        ]

    # Finish combos
    if "finish" in word_set and "thank you" in word_set:
        return [
            "I am finished, thank you.",
            "I am done, thank you very much.",
            "That is all, thank you.",
        ]
    if "finish" in word_set and "food" in word_set:
        return [
            "I have finished my food.",
            "I am done eating.",
            "I finished the food.",
        ]

    return None   # no exact rule matched


# ── Priority 2: Intent-pair rules ─────────────────────────────────────────────
def _pair_rules(word_set, intents, obj):
    o = obj or "that"
    if "gratitude" in intents and "object" in intents:
        return [
            f"Thank you for the {o}.",
            f"Thank you, I appreciate the {o}.",
            f"Thank you, the {o} was great.",
        ]
    if "greeting" in intents and "object" in intents:
        return [
            f"Hello, I want {o}.",
            f"Hello, can I have {o}?",
            f"Hello, I need {o}.",
        ]
    if "request" in intents and "object" in intents:
        return [
            f"Please give me {o}.",
            f"I want {o}, please.",
            f"Can I have {o}?",
        ]
    if "positive" in intents and "object" in intents:
        return [
            f"Yes, I want {o}.",
            f"Yes, please give me {o}.",
            f"Yes, I would like {o}.",
        ]
    if "negative" in intents and "object" in intents:
        return [
            f"No, I do not want {o}.",
            f"No {o}, thank you.",
            f"I do not need {o}.",
        ]
    if "hunger" in intents and "object" in intents:
        return [
            f"I am hungry and I want {o}.",
            f"I am hungry, can I have {o}?",
            f"Please give me {o}, I am hungry.",
        ]
    return []


# ── Priority 3: Single-intent rules ──────────────────────────────────────────
def _single_rules(intents, obj):
    o = obj or "that"
    suggestions = []
    if "gratitude"  in intents: suggestions += ["Thank you!", "Thank you very much.", "Thanks!"]
    if "greeting"   in intents: suggestions += ["Hello! How are you?", "Hello!", "Hi there!"]
    if "positive"   in intents: suggestions += ["Yes.", "Yes, please.", "Yes, I agree."]
    if "negative"   in intents: suggestions += ["No.", "No, thank you.", "No, I am fine."]
    if "request"    in intents: suggestions += ["Please help me.", "I need help.", "Can you help me?"]
    if "hunger"     in intents: suggestions += ["I am hungry.", "I want to eat.", "Can I have food?"]
    if "finish"     in intents: suggestions += ["I am done.", "I have finished.", "That is all."]
    if "object"     in intents: suggestions += [f"I want {o}.", f"Can I have {o}?", f"Please give me {o}."]
    return suggestions


# ── Public API ────────────────────────────────────────────────────────────────
def generate_sentences(words):
    """
    Input : list of detected words  e.g. ["thank you", "water"]
    Output: list of 2-3 ranked sentence suggestions.
    """
    if not words:
        return []

    words    = _deduplicate(words)
    word_set = set(words)
    intents  = _intents(words)
    obj      = _obj(words)

    result = _exact_rules(word_set)
    if result:
        return _fmt(result)

    result = _pair_rules(word_set, intents, obj)
    if result:
        return _fmt(result)

    result = _single_rules(intents, obj)
    if result:
        return _fmt(result)

    joined = " ".join(words)
    return _fmt([
        f"I am trying to say: {joined}.",
        f"I mean: {joined}.",
        f"Please understand: {joined}.",
    ])