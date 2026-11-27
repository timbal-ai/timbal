"""Reply language for a voice turn.

``VoiceConfig.language`` used to reach only the speech-to-text adapter. The
text model then saw whatever the agent prompt said, often a soft "match the
caller's language", and some models answer an English transcript in another
language. Production voice stacks pin a configured language in the LLM
instructions (LiveKit ``update_instructions``, OpenAI Realtime language
policy) and treat an unset language as "the latest substantive user
utterance", not a vague caller-language guess.

The instruction is English on purpose: it tells the model which language to
speak. It does not hardcode a reply language. Unknown BCP-47 tags are named
by the tag itself.
"""

from __future__ import annotations

import re

from .config import language_key

# Primary subtag plus up to three subtags. Tight enough that a client-set
# ``language`` (it is on the voice hello allow-list) cannot carry a newline
# or a prompt into the system message.
_LANGUAGE_TAG = re.compile(r"^[a-z]{2,8}(?:-[0-9A-Za-z]{1,8}){0,3}$")

# ISO 639-1 primary subtags we are willing to name. Anything else is still
# pinned, using the tag as the name, so a language missing from this map is
# not treated as English.
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "eu": "Basque",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nb": "Norwegian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_UNSET_REPLY_LANGUAGE = (
    "If your instructions above do not already name a single reply language, reply in the "
    "language of the user's latest substantive message. A substantive message is a complete "
    "request, question, or correction. Do not switch because of accent, filler sounds, "
    "backchannels, names, or an isolated word from another language. When that latest message "
    "is in another language, switch to it and stay there. If the user explicitly asks to use "
    "a language, use that one."
)


def normalize_language_tag(code: str | None) -> str | None:
    """``" ES_es "`` → ``"es-es"`` primary lowercased; blank → ``None``.

    Region subtags are kept (``"en-US"`` stays ``"en-US"``) so a provider hint
    and the reply instruction name the same tag. A value that is not a BCP-47
    tag returns ``None`` rather than being copied into the system prompt.
    """
    if code is None:
        return None
    raw = str(code).strip().replace("_", "-")
    if not raw:
        return None
    primary, sep, rest = raw.partition("-")
    if not primary.isalpha():
        return None
    primary = primary.lower()
    tag = primary if not sep else f"{primary}-{rest}"
    if not _LANGUAGE_TAG.fullmatch(tag):
        return None
    return tag


def reply_language_instruction(language: str | None) -> str:
    """The reply-language paragraph for one voice turn.

    A configured tag is a firm pin (``"Reply in Spanish (es)."``). ``None``
    matches the latest substantive user message and does not name a default
    language. Explicit user requests to switch are honored either way; accent,
    fillers, and a single borrowed word are not.
    """
    tag = normalize_language_tag(language)
    if not tag:
        return _UNSET_REPLY_LANGUAGE
    name = _LANGUAGE_NAMES.get(language_key(tag), tag)
    return (
        f"Reply in {name} ({tag}). This is the conversation language for every spoken line, "
        "including short replies and waiting phrases. Do not switch because of accent, filler "
        "sounds, backchannels, names, or an isolated word from another language. Switch only "
        "when the user explicitly asks to use another language."
    )


def with_reply_language(system_prompt: str | None, language: str | None) -> str:
    """``system_prompt`` plus :func:`reply_language_instruction`, instruction last.

    The pin has to follow the agent's own prompt. A soft "match the caller's
    language" earlier in the prompt is what models were ignoring.
    """
    policy = reply_language_instruction(language)
    base = (system_prompt or "").rstrip()
    if not base:
        return policy
    return f"{base}\n\n{policy}"
