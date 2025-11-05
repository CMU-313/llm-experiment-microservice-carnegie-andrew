import os
import re
from ollama import Client

# Get OLLAMA_HOST, if specified, or default to localhost:11434.
# Using tinyllama as it's small enough to run in limited memory environments
MODEL_NAME = os.getenv("OLLAMA_MODEL", "tinyllama")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Initialize the Ollama client
client = Client(host=OLLAMA_URL)

# --- Default contexts ---

DEFAULT_TRANSLATION_CONTEXT = """You are a translation assistant.

Rules:
- Translate input to English.
- If input is already English, return it unchanged.
- Output ONLY the final English text: no labels (e.g., "English:"), no explanations, no quotes, no brackets, no markdown.
- Preserve meaning, tone, punctuation, numbers, URLs, usernames, and code.
- Do not add or remove information.

Good examples:
INPUT: Hier ist dein erstes Beispiel.
OUTPUT: Here is your first example.

INPUT: Please review the attached document.
OUTPUT: Please review the attached document.
"""

DEFAULT_CLASSIFICATION_CONTEXT = """You are a language classifier.

Rules:
- Output EXACTLY ONE label chosen from this list: English, German, French, Spanish, Portuguese, Italian, Chinese, Japanese, Korean, Arabic, Russian, Turkish, Vietnamese, Hindi, Indonesian, Greek.
- If uncertain, output EXACTLY: Unknown
- No explanations, no markdown, no quotes, no ISO codes, no extra words.

Good examples:
INPUT: ¿Podrías ayudarme?
OUTPUT: Spanish

INPUT: これは日本語ですか。
OUTPUT: Japanese

INPUT: I ain't seeing that button.
OUTPUT: English

"""
_LABELS = [
    "English","German","French","Spanish","Portuguese","Italian",
    "Chinese","Japanese","Korean","Arabic","Russian","Turkish",
    "Vietnamese","Hindi","Indonesian","Greek","Unknown"
]
_LABEL_RE = re.compile(r"\b(" + "|".join(l.lower() for l in _LABELS) + r")\b", re.I)

def _extract_label(text: str) -> str:
    if not text:
        return "Unknown"
    m = _LABEL_RE.search(text.strip())
    return m.group(1).title() if m else "Unknown"
def get_language(post: str) -> str:
    """Detect the language of the input text using the LLM."""
    system_ctx = DEFAULT_CLASSIFICATION_CONTEXT

    messages = [
        {"role": "system", "content": system_ctx},
        # Few-shot to anchor ONE-word, English labels only
        {"role": "user", "content": "INPUT: Das ist ein kurzer Test.\nOUTPUT:"},
        {"role": "assistant", "content": "German"},
        {"role": "user", "content": "INPUT: Bonjour, comment ça va ?\nOUTPUT:"},
        {"role": "assistant", "content": "French"},
        {"role": "user", "content": "INPUT: I ain't seeing that button on my screen.\nOUTPUT:"},
        {"role": "assistant", "content": "English"},

        # Actual request with closed label reminder
        {"role": "user", "content":
            ("Task: Detect the language of the following text.\n"
             "Choose EXACTLY ONE from: English, German, French, Spanish, Portuguese, Italian, "
             "Chinese, Japanese, Korean, Arabic, Russian, Turkish, Vietnamese, Hindi, Indonesian, Greek.\n"
             "If uncertain, output: Unknown\n\n"
             f"INPUT:\n{post}\n\nOUTPUT:")}
    ]
    try:
        resp = client.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": 0.0, "top_p": 0.1}
        )
        raw = (getattr(resp, "message", None) and resp.message.content) or ""
        label = _extract_label(raw)
        return label
    except Exception:
        return "Unknown"


def get_translation(post: str) -> str:
    """Translate the input text to English using the LLM."""
    system_ctx = DEFAULT_TRANSLATION_CONTEXT

    messages = [
        {"role": "system", "content": system_ctx},
        {"role": "user", "content": "INPUT: Hier ist dein erstes Beispiel.\nOUTPUT:"},
        {"role": "assistant", "content": "Here is your first example."},
        {"role": "user", "content": "INPUT: Please review the attached document.\nOUTPUT:"},
        {"role": "assistant", "content": "Please review the attached document."},
        {"role": "user", "content":
            ("Task: Translate the text into English.\n"
             "- If the input is already English, return it unchanged.\n"
             "- Output ONLY the final English text. Do not write phrases like "
             "'the translation is', 'in English', 'English:', 'Translation:'. "
             "Do not include quotes or any extra words.\n"
             "- Preserve meaning, tone, punctuation, numbers, URLs, usernames, and code.\n\n"
             f"INPUT:\n{post}\n\nOUTPUT:")}
    ]

    try:
        resp = client.chat(model=MODEL_NAME, messages=messages,
                           options={"temperature": 0.0, "top_p": 0.1})
        raw = (getattr(resp, "message", None) and resp.message.content) or ""
        cleaned = raw.strip()
        return cleaned if cleaned else post
    except Exception:
        return post

def query_llm(post: str) -> tuple[bool, str]:
    """
    Returns (is_english, english_text).
    - If English: (True, original)
    - Else: (False, translation)
    Robust to model drift and failures.
    """
    try:
        lang = get_language(post) or "Unknown"
        is_eng = "english" in lang.lower()

        if is_eng:
            return (True, post.strip())

        translated = (get_translation(post) or "").strip()
        if not translated:
            translated = post.strip()  # safe fallback
        return (False, translated)
    except Exception:
        return (False, (post or "").strip())
def translate_content(content: str) -> tuple[bool, str]:
    return query_llm(content)