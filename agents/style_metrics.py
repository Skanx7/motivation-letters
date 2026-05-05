"""Deterministic style metrics. No LLM calls. No external libraries.

These are the numbers AI detectors actually compute. The MetricsCritic uses
them to score how closely a draft matches the user's example letters on:
  - sentence-length variance (burstiness)
  - average sentence length
  - lexical diversity (type/token ratio)
  - average word length
  - comma rate per sentence
  - em-dash count
  - parenthesis count
  - n-gram overlap with the user's own writing

The baseline (mean + std per metric) is computed once from the example letters
at startup and reused for every draft scored thereafter."""

import re
import statistics

# Word characters including common French/Spanish accents.
WORD_RE = re.compile(r"[\wÀ-ſ]+", re.UNICODE)
SENT_END = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


# --- individual metrics ------------------------------------------------------

def sentence_length_variance(text: str) -> float:
    """Variance of sentence length in words. Real writing has higher variance
    than LLM output; LLMs cluster sentences around the same length."""
    sents = split_sentences(text)
    lens = [len(tokenize(s)) for s in sents]
    if len(lens) < 2:
        return 0.0
    return statistics.pvariance(lens)


def avg_sentence_length(text: str) -> float:
    sents = split_sentences(text)
    lens = [len(tokenize(s)) for s in sents]
    return statistics.mean(lens) if lens else 0.0


def lexical_diversity(text: str) -> float:
    """Type/token ratio. 1.0 = every word unique; lower = repetitive."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def avg_word_length(text: str) -> float:
    tokens = tokenize(text)
    return statistics.mean(len(t) for t in tokens) if tokens else 0.0


def comma_rate(text: str) -> float:
    """Commas per sentence. LLMs over-comma."""
    sents = split_sentences(text)
    if not sents:
        return 0.0
    return text.count(",") / len(sents)


def em_dash_count(text: str) -> int:
    return text.count("—") + text.count("–")


def paren_count(text: str) -> int:
    return text.count("(")


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_overlap(text: str, reference_corpus: str, n: int = 2) -> float:
    """Fraction of draft n-grams that also appear in reference_corpus.
    High overlap means the draft uses the user's vocabulary; low overlap
    means it sounds like generic LLM phrasing."""
    draft_grams = _ngrams(tokenize(text), n)
    if not draft_grams:
        return 0.0
    ref_grams = set(_ngrams(tokenize(reference_corpus), n))
    if not ref_grams:
        return 0.0
    return sum(1 for g in draft_grams if g in ref_grams) / len(draft_grams)


# --- aggregate ---------------------------------------------------------------

def compute_metrics(text: str, reference_corpus: str = "") -> dict[str, float]:
    m: dict[str, float] = {
        "sentence_length_variance": sentence_length_variance(text),
        "avg_sentence_length": avg_sentence_length(text),
        "lexical_diversity": lexical_diversity(text),
        "avg_word_length": avg_word_length(text),
        "comma_rate": comma_rate(text),
        "em_dash_count": float(em_dash_count(text)),
        "paren_count": float(paren_count(text)),
    }
    if reference_corpus:
        m["bigram_overlap"] = ngram_overlap(text, reference_corpus, 2)
        m["trigram_overlap"] = ngram_overlap(text, reference_corpus, 3)
    return m


def compute_baseline(example_texts: list[str]) -> dict[str, dict[str, float]]:
    """Returns {metric_name: {mean, std}} computed across the example letters.
    With only one example, std falls back to a generous fraction of the mean
    so the score isn't impossibly tight."""
    if not example_texts:
        return {}
    per = [compute_metrics(t) for t in example_texts]
    out: dict[str, dict[str, float]] = {}
    keys: set[str] = set()
    for m in per:
        keys.update(m.keys())
    for name in keys:
        vals = [m[name] for m in per if name in m]
        if not vals:
            continue
        mean = statistics.mean(vals)
        if len(vals) > 1:
            std = statistics.pstdev(vals)
        else:
            std = max(abs(mean) * 0.25, 0.5)
        out[name] = {"mean": mean, "std": max(std, 0.1)}
    return out


# --- scoring helpers ---------------------------------------------------------

def score_against_baseline(value: float, baseline: dict[str, float] | None) -> int:
    """Map a value to a 1-10 score by z-distance to the baseline mean.
    Within 0.5 std = 10, 1.0 = 9, 1.5 = 8, ..., beyond 4 = 1."""
    if not baseline:
        return 5
    mean = baseline.get("mean", 0.0)
    std = max(baseline.get("std", 0.1), 0.1)
    z = abs(value - mean) / std
    if z <= 0.5:
        return 10
    if z <= 1.0:
        return 9
    if z <= 1.5:
        return 8
    if z <= 2.0:
        return 7
    if z <= 2.5:
        return 6
    if z <= 3.0:
        return 5
    if z <= 3.5:
        return 4
    if z <= 4.0:
        return 3
    if z <= 5.0:
        return 2
    return 1


def score_ngram_overlap(fraction: float) -> int:
    """Higher overlap = better. Calibrated for short user corpora (1-3 letters)
    where 15-25% overlap is realistic for the user's vocabulary footprint."""
    pct = fraction * 100
    if pct >= 25:
        return 10
    if pct >= 20:
        return 9
    if pct >= 15:
        return 8
    if pct >= 12:
        return 7
    if pct >= 9:
        return 6
    if pct >= 6:
        return 5
    if pct >= 4:
        return 4
    if pct >= 2:
        return 3
    if pct >= 1:
        return 2
    return 1
