"""Approximate text predictability via Ollama's logprobs on echo.

The trick: ask the model to repeat the text VERBATIM with logprobs=true,
then read the per-token logprobs of its (faithful) echo. For each token
that's a copy of the input, the logprob reflects how surprising that token
was given the preceding context. Low surprise = AI-like (the model would
have written this anyway); high surprise = human-like.

Caveat: this is NOT true text perplexity (which would require scoring the
text under the model's prior, not under an echo prompt). But it tracks the
same signal AI detectors look at - the model's "comfort" with the text -
and it's the only way to get logprobs from Ollama's OpenAI-compat endpoint
without accessing native llama.cpp logits.

Returns None whenever logprobs are unavailable on the model or endpoint, so
the MetricsCritic can fall back to a neutral score."""

import math
import statistics
import sys

from openai import OpenAI

from llm import chat_extra

ECHO_SYSTEM = (
    "Repeat the user's message VERBATIM. Do not add anything. "
    "Do not remove anything. Output only the repeated text."
)


def _call_with_logprobs(client: OpenAI, model: str, text: str) -> list[float] | None:
    text = text.strip()
    if len(text) < 50:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ECHO_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_tokens=int(len(text.split()) * 4) + 100,
            temperature=0.0,
            logprobs=True,
            top_logprobs=1,
            **chat_extra(),
        )
    except Exception as e:
        print(f"[perplexity] echo call failed: {e.__class__.__name__}: {e}", file=sys.stderr)
        return None

    lp = getattr(resp.choices[0], "logprobs", None)
    if lp is None:
        return None
    content = getattr(lp, "content", None)
    if not content:
        return None
    out = [t.logprob for t in content if getattr(t, "logprob", None) is not None]
    return out or None


def echo_perplexity(client: OpenAI, model: str, text: str) -> float | None:
    """Approximate perplexity = exp(-mean_logprob). Returns None on failure."""
    logprobs = _call_with_logprobs(client, model, text)
    if not logprobs:
        return None
    avg = sum(logprobs) / len(logprobs)
    try:
        return math.exp(-avg)
    except OverflowError:
        return None


def baseline_perplexity_from_examples(client: OpenAI, model: str, examples: list[str]) -> float | None:
    """Average perplexity across the example letters. Used as the human-target
    perplexity that a draft should match or exceed."""
    vals: list[float] = []
    for ex in examples:
        p = echo_perplexity(client, model, ex)
        if p is not None and p > 0:
            vals.append(p)
    if not vals:
        return None
    return statistics.mean(vals)


def score_predictability(draft_perplexity: float | None, baseline_perplexity: float | None) -> int:
    """Higher draft perplexity (relative to the example letters' baseline) means
    the model finds the text less predictable, which is what we want.
    A draft perplexity at or above baseline scores well."""
    if draft_perplexity is None or baseline_perplexity is None or baseline_perplexity <= 0:
        return 5  # neutral when measurement isn't available
    ratio = draft_perplexity / baseline_perplexity
    if ratio >= 1.0:
        return 10
    if ratio >= 0.9:
        return 9
    if ratio >= 0.8:
        return 8
    if ratio >= 0.7:
        return 7
    if ratio >= 0.6:
        return 6
    if ratio >= 0.5:
        return 5
    if ratio >= 0.4:
        return 4
    if ratio >= 0.3:
        return 3
    if ratio >= 0.2:
        return 2
    return 1
