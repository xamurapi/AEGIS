"""Versioned prompt templates (spec M8.2).

Prompts live in ``.md`` files next to this module rather than inline in the
code that sends them, for one reason that matters: a reasoning strategy (M6) is
persisted data that names a template, and the arena compares strategies across
runs. If the text a template resolves to changed silently with a code edit,
every historical win-rate would be measuring a prompt that no longer exists.

Files are named ``<name>.md``. Substitution is ``{placeholder}`` and is
deliberately not Python formatting: a template containing a JSON example is
full of braces, and ``str.format`` would choke on every one of them.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("aegis.cortex.prompts")

PROMPT_DIR = Path(__file__).resolve().parent
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class PromptError(Exception):
    """A template was requested that does not exist."""


@lru_cache(maxsize=128)
def load(name: str) -> str:
    """Raw template text. Cached — templates do not change at runtime."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise PromptError(f"unknown prompt template {name!r}; available: {available()}")
    return path.read_text(encoding="utf-8")


def available() -> list[str]:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.md"))


def render(name: str, **values) -> str:
    """Fill a template's placeholders.

    Only lowercase ``{snake_case}`` tokens are substituted, so the JSON braces
    that most of these templates contain pass through untouched. A placeholder
    with no supplied value is left as-is and logged: a template silently
    rendering "None" into a prompt is far harder to notice than one that still
    shows its own placeholder.
    """
    text = load(name)
    missing = []

    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        missing.append(key)
        return match.group(0)

    rendered = _PLACEHOLDER.sub(substitute, text)
    if missing:
        logger.warning("Prompt %s rendered without values for: %s",
                       name, sorted(set(missing)))
    return rendered


def version(name: str) -> str:
    """Content hash of a template — the identity a stored strategy refers to."""
    return hashlib.blake2b(load(name).encode("utf-8"), digest_size=8).hexdigest()


def versions() -> dict[str, str]:
    return {name: version(name) for name in available()}
