#!/usr/bin/env python3
"""
lexicon — the words YOU use, learned rather than typed.

Whisper hears a word better when it has seen it recently: `initial_prompt` is
treated as preceding transcript, so listing your vocabulary biases the decoder
toward hearing those words. That slot is the single highest-leverage thing in
this app — it is why Whisper heard "VWAP" and "IBKR" in the A/B where Parakeet,
which has no prompt, heard "voir" and "Ibka".

But the slot is SMALL and it truncates silently:

    decoder context = 448 tokens · initial_prompt is capped at HALF of that
    -> 224 tokens (~150 words), and mlx_whisper keeps only the LAST 224
       (decoding.py: prompt_tokens[-(self.n_ctx // 2 - 1):])

So a lexicon that just grows is a lexicon that silently loses its head. The job
is not "collect words" — it is "rank words and spend 224 tokens well."

WHAT EARNS A SLOT
    Words the decoder is likely to get WRONG and that you actually say:
    acronyms (VWAP, IBKR, UTMB), tickers (NVDA), proper nouns (Streamlit), and
    domain jargon (avwap, scalp). Ordinary English is dropped — "the" and
    "bother" need no help, and every ordinary word you list steals a slot from
    one that does.

WHERE THE WORDS COME FROM
    seed:  your own writing. Your notes already contain your vocabulary,
           spelled correctly, in your terms. This is the fix for the
           chicken-and-egg problem — a dictation log can never teach the app a
           word it has never once transcribed correctly.
    grow:  your dictation log, once running. New terms surface as you use them.

    usage:  ./lexicon.py            rebuild from corpus + log, show the result
            ./lexicon.py --show     print the current lexicon and its cost
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
LEXICON_FILE = HERE / "lexicon.json"
DICTATION_LOG = HERE / "dictations.log"

# Your writing — where the lexicon is seeded from. Globs, newest content wins.
CORPUS_ROOT = Path.home() / "Coding"
CORPUS_GLOBS = ["*.md", "trading/**/*.md", "learning/**/*.md",
                "investing/**/*.md", "day_trading_market_analysis/**/*.md",
                "reference/**/*.md"]

# Whisper truncates the prompt to the LAST 224 tokens. Stay well under: the
# glossary preamble costs a few, and a token is not a word (acronyms often
# cost 2-3). ~1.6 tokens/word is a safe estimate for jargon-heavy vocab.
TOKEN_BUDGET = 180
TOKENS_PER_WORD = 1.6
MAX_WORDS = int(TOKEN_BUDGET / TOKENS_PER_WORD)   # ~112 words

MIN_COUNT = 3          # say it three times before it earns a slot
SYSTEM_DICT = Path("/usr/share/dict/words")

# Function words. These are the trap: you SHOUT them in markdown for emphasis
# ("the card wins ALL conflicts", "do NOT trade past the fuse box"), and an
# acronym regex cannot tell that shout from a real acronym. Whisper needs no
# help hearing "the" — every one of these that reaches the glossary steals a
# slot from a word that does. Blocked regardless of case.
STOPWORDS = set("""
a an and any are as at be been but by can cant do does doesnt doing done dont
for from get gets go goes going had has have how i if in into is it its just
like make makes many may me more most much must my no not now of off on once
one only or other our out over own said same see should so some such take than
that the their them then there these they this those to too up us use used
uses very wait want was way we were what when where which while who why will
with would you your yours new old first last next each both all also any every
buy sell long short stop start end open close high low big small good bad best
jan feb mar apr may jun jul aug sep oct nov dec mon tue wed thu fri sat sun
held paid made kept told found built came went knew left felt sent meant
apps hype entries copies committed
""".split())


def ordinary_english():
    """The words that need no help. Anything Whisper already knows how to hear
    is a wasted slot. The system dictionary holds only BASE forms, so a plural
    or a participle ("names", "closes", "packaged") looks exotic to it and
    sneaks in — de-inflect before asking."""
    if not SYSTEM_DICT.exists():
        return set()
    return {w.strip().lower() for w in SYSTEM_DICT.read_text(
        errors="ignore").splitlines() if w.strip()}


def is_ordinary(word, dictionary):
    """True if this is just an English word wearing a costume."""
    w = word.lower()
    if w in STOPWORDS:
        return True
    if w in dictionary:
        return True
    # de-inflect: names->name, closes->close, packaged->package, using->use,
    # copies->copy. The system dictionary holds base forms only, so without
    # this every plural in your notes looks like exotic jargon.
    for suffix, stems in (("s", [""]), ("es", ["", "e"]), ("ies", ["y"]),
                          ("ed", ["", "e"]), ("ing", ["", "e"]),
                          ("ly", [""]), ("er", ["", "e"])):
        if w.endswith(suffix):
            root = w[: -len(suffix)]
            if any((root + s) in dictionary for s in stems):
                return True
    return False


def candidates(text):
    """Terms worth biasing the decoder toward, in the form YOU write them.

    Three shapes, all of them things a general ASR model mishears: ACRONYMS
    (VWAP), Proper-Nouns (Streamlit), and lowercase jargon that isn't a word
    (avwap, preprep). Everything is filtered by is_ordinary() afterwards."""
    out = []
    # ACRONYMS / tickers: 2-6 caps, optionally with digits — VWAP, NVDA, ATR
    out += re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", text)
    # Proper nouns: Capitalised, not at the start of a line/sentence
    out += re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", text, re.M)
    # lowercase jargon. \b...\b would split "doesn't" into "doesn" — require
    # the word not be followed by an apostrophe, so contractions stay whole.
    out += re.findall(r"\b[a-z][a-z-]{3,}\b(?!')", text)
    return out


def harvest(dictionary):
    """Count every candidate term across your corpus + your dictation log."""
    counts = Counter()
    files = []
    for pattern in CORPUS_GLOBS:
        files += list(CORPUS_ROOT.glob(pattern))
    if DICTATION_LOG.exists():
        files.append(DICTATION_LOG)

    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        # strip code/paths/urls — they are full of tokens you never SPEAK
        text = re.sub(r"`[^`]*`|```.*?```|https?://\S+|[\w./-]*/[\w./-]+",
                      " ", text, flags=re.S)
        for w in candidates(text):
            if not is_ordinary(w, dictionary):
                counts[w] += 1
    return counts, len(files)


def build():
    """Rank the harvest, spend the 224-token budget, write lexicon.json."""
    ordinary = ordinary_english()
    counts, n_files = harvest(ordinary)

    # collapse case variants onto the spelling you use most (vwap/VWAP -> VWAP)
    best = {}
    for word, n in counts.items():
        key = word.lower()
        if key not in best or n > counts[best[key]]:
            best[key] = word
    ranked = sorted(
        ((best[k], sum(c for w, c in counts.items() if w.lower() == k))
         for k in best),
        key=lambda wc: -wc[1],
    )
    ranked = [(w, n) for w, n in ranked if n >= MIN_COUNT][:MAX_WORDS]

    lex = {
        "words": [w for w, _ in ranked],
        "counts": {w: n for w, n in ranked},
        "sources": n_files,
        "budget": {"max_words": MAX_WORDS, "token_budget": TOKEN_BUDGET},
    }
    LEXICON_FILE.write_text(json.dumps(lex, indent=2))
    return lex


def words():
    """The current lexicon (built on first use)."""
    if not LEXICON_FILE.exists():
        return build()["words"]
    return json.loads(LEXICON_FILE.read_text())["words"]


def glossary_prompt():
    """The string Whisper actually sees. Phrased as prose, not a list: the
    prompt is treated as PRECEDING TRANSCRIPT, so it should read like speech."""
    w = words()
    return "Glossary: " + ", ".join(w) + "." if w else None


def log_dictation(text):
    """Append a transcript so tomorrow's lexicon knows today's words. Local,
    plain text, yours to delete — nothing here ever leaves the machine."""
    try:
        with DICTATION_LOG.open("a") as f:
            f.write(text.replace("\n", " ") + "\n")
    except Exception:
        pass   # logging must never break a dictation


if __name__ == "__main__":
    if "--show" in sys.argv and LEXICON_FILE.exists():
        lex = json.loads(LEXICON_FILE.read_text())
    else:
        lex = build()
    w = lex["words"]
    est = int(len(w) * TOKENS_PER_WORD)
    print(f"lexicon · {len(w)} words · ~{est} of {TOKEN_BUDGET} tokens "
          f"(hard cap 224) · from {lex['sources']} files\n")
    for word in w:
        print(f"  {lex['counts'][word]:>5}x  {word}")
