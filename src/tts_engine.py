"""
tts_engine.py
Text-to-speech wrapper.

Uses pyttsx3 (offline, no internet needed).
Falls back silently if the library is not installed so the rest of
the system continues to work without it.

Install:  pip install pyttsx3
"""

_engine = None
_available = False


def _init():
    global _engine, _available
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 155)    # words per minute (default ~200)
        _engine.setProperty("volume", 1.0)
        _available = True
    except Exception as e:
        print(f"[TTS] pyttsx3 not available: {e}")
        print("[TTS] Install with:  pip install pyttsx3")
        _available = False


_init()


def speak(text: str):
    """
    Speak `text` asynchronously (non-blocking).
    If TTS is unavailable, does nothing and prints the text instead.
    """
    if not text:
        return
    if _available and _engine is not None:
        try:
            _engine.say(text)
            _engine.runAndWait()
        except Exception as e:
            print(f"[TTS] speak error: {e}")
    else:
        print(f"[TTS] Would say: {text}")


def is_available() -> bool:
    return _available