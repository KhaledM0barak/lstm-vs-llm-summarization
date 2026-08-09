"""Word-level tokenizer.

Deliberately simple and self-contained: the assignment asks us to own the
preprocessing, and a transparent regex tokenizer also makes the out-of-vocabulary
failure mode (a central part of the error analysis) directly observable, which a
subword tokenizer would hide by construction.
"""

from __future__ import annotations

import re

# Words, numbers with internal separators, and standalone punctuation.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-.,:/][a-z0-9]+)*|[^\sa-z0-9]")

# CNN/DailyMail articles carry a wire-service preamble and a trailing byline
# boilerplate block; both are noise that would otherwise consume the encoder's
# limited 400-token window.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:\(cnn\)\s*(?:--)?|by\s*\.\s*[^.]{0,80}?\s*\.\s*published\s*:.*?\|\s*)",
    re.IGNORECASE | re.DOTALL,
)
_BOILERPLATE_RE = re.compile(
    r"@highlight|last updated at .{0,40}?on \d+ \w+ \d{4}",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip dataset boilerplate, and collapse whitespace."""
    text = text.replace(" ", " ").strip().lower()
    text = _PREAMBLE_RE.sub("", text)
    text = _BOILERPLATE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalize then split into word-level tokens."""
    return _TOKEN_RE.findall(normalize(text))


# Sentence splitting runs on lowercased text, so capitalization is unavailable as
# a boundary cue. Without protection, "the u.s. said" splits into two sentences,
# which shortens Lead-3 and corrupts the sentence union that ROUGE-Lsum computes.
# Only abbreviations that are unlikely to also be a sentence-final English word.
# Deliberately excluded: "no" (No. 5 vs. "he said no."), "min"/"max"/"est"/"co"/
# "act"/"sun" and similar -- each is far more often an ordinary word ending a
# sentence than an abbreviation, and a false merge silently joins two sentences,
# which shortens Lead-3 and corrupts the sentence union ROUGE-Lsum computes.
# Missing a genuine "No. 5" split costs far less than that.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "gen", "sen",
    "rep", "gov", "lt", "sgt", "col", "capt", "cmdr", "adm", "maj", "supt", "det",
    "inc", "ltd", "corp", "vs", "etc", "eg", "ie", "vol", "approx", "dept", "fig",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "a.m", "p.m",
}
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# A single letter ("j") or a dotted initialism ("u.s", "f.b.i") as it appears
# with its trailing period already stripped.
_INITIALISM_RE = re.compile(r"(?:[a-z]\.)*[a-z]")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, without breaking inside common abbreviations.

    Also refuses to split after a single *letter*, which covers personal initials
    and dotted initialisms ("j. smith", "u.s.") that survive normalization as
    separate letter-dot pairs.

    The single-character rule deliberately excludes digits. Applying it to digits
    swallows sentence-final numbers -- "it happened on jan. 4. everyone saw."
    would collapse into one sentence, because "jan." merges as an abbreviation
    and then "4." merges as a single character. Dates ending a sentence are
    common in news text, and the result was a silently shorter Lead-3 and a
    corrupted sentence union for ROUGE-Lsum.
    """
    pieces = _SENT_SPLIT_RE.split(text.strip())
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if out:
            tail = out[-1].rstrip()
            words = tail[:-1].split() if tail.endswith(".") else []
            last_word = words[-1].lower() if words else ""
            # Single letters and dotted initialisms: "j." (an initial), "u.s.",
            # "f.b.i.". Requiring letters excludes "4.", which must terminate.
            is_initialism = bool(_INITIALISM_RE.fullmatch(last_word))
            if tail.endswith(".") and (last_word in _ABBREVIATIONS or is_initialism):
                out[-1] = tail + " " + piece
                continue
        out.append(piece)
    return [s for s in out if s.strip()]


def detokenize(tokens: list[str]) -> str:
    """Rejoin tokens into readable text.

    Used only for display and for ROUGE input; it is a best-effort inverse, since
    lowercasing and whitespace collapsing are not invertible.
    """
    out: list[str] = []
    for tok in tokens:
        if out and (tok in ".,!?;:%)]}'" or tok == "n't"):
            out[-1] = out[-1] + tok
        elif out and out[-1] in "([{$":
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return " ".join(out)
