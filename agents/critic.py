"""A panel of specialist critics, not one overloaded generalist.

Splitting the critic into focused specialists has two benefits:
1. Each prompt is short and on-task, so the model doesn't get lost juggling
   8 axes and a hundred rules.
2. The writer's revision step gets feedback grouped by concern, so it can
   target the weakest specialist's complaints first instead of trying to
   fix everything everywhere at once.

Specialists:
- StyleCritic         : style_fit, language, conversationality
- AntiAICritic        : no_ai_tells, no_company_recap
- AuthenticityCritic  : authenticity, job_relevance
- SubstanceCritic     : thinking

CriticPanel runs all four (in parallel) and merges per-axis scores into a
single Critique with the union of strengths/weaknesses/suggestions, plus the
ship verdict gated on the MIN across all 8 axes."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from openai import OpenAI

from llm import chat_extra

from .perplexity import echo_perplexity, score_predictability
from .style_metrics import compute_metrics, score_against_baseline, score_ngram_overlap

# LLM-judged axes (gated strictly at 9 for ship)
LLM_AXES = [
    "job_relevance",
    "authenticity",
    "style_fit",
    "language",
    "thinking",
    "conversationality",
    "no_ai_tells",
    "no_company_recap",
]

# Deterministic metric axes (gated more leniently - they're noisy distribution
# signals, a perfect 9-10 across all 5 is rare even on real human writing)
METRIC_AXES = [
    "burstiness_match",
    "lexical_diversity_match",
    "punctuation_match",
    "ngram_authenticity",
    "predictability",
]

ALL_AXES = LLM_AXES + METRIC_AXES

SHIP_PER_AXIS_THRESHOLD = 9         # default for LLM axes
METRIC_SHIP_THRESHOLD = 6           # relaxed for metric axes
GATING_THRESHOLDS: dict[str, int] = {
    **{a: SHIP_PER_AXIS_THRESHOLD for a in LLM_AXES},
    **{a: METRIC_SHIP_THRESHOLD for a in METRIC_AXES},
}


# --- shared dataclasses ------------------------------------------------------

@dataclass
class SpecialistVerdict:
    """One specialist critic's output."""
    name: str
    scores: dict[str, int] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Critique:
    """Aggregated critique from the whole panel."""
    scores: dict[str, int] = field(default_factory=dict)
    verdict: str = "revise"
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    by_critic: dict[str, SpecialistVerdict] = field(default_factory=dict)

    @property
    def min_score(self) -> int:
        return min(self.scores.values()) if self.scores else 0

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def score(self) -> int:
        """Gating metric for the convergence loop. MIN across axes ensures every
        grade matters - one weak axis blocks ship."""
        return self.min_score

    @property
    def bottleneck_axis(self) -> str:
        if not self.scores:
            return ""
        return min(self.scores, key=self.scores.__getitem__)

    @property
    def bottleneck_critic(self) -> str:
        """Which specialist's worst axis is the bottleneck."""
        axis = self.bottleneck_axis
        for name, sv in self.by_critic.items():
            if axis in sv.scores:
                return name
        return ""

    def to_dict(self) -> dict:
        return {
            "scores": dict(self.scores),
            "min_score": self.min_score,
            "avg_score": round(self.avg_score, 2),
            "bottleneck_axis": self.bottleneck_axis,
            "bottleneck_critic": self.bottleneck_critic,
            "verdict": self.verdict,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "by_critic": {
                name: {"scores": sv.scores, "weaknesses": sv.weaknesses, "suggestions": sv.suggestions}
                for name, sv in self.by_critic.items()
            },
        }


# --- JSON parsing helpers (shared by all specialists) ------------------------

def _repair_truncated_json(text: str) -> str:
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    repaired = text
    if in_string:
        repaired += '"'
    for ch in reversed(stack):
        repaired += "}" if ch == "{" else "]"
    return repaired


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith(("json\n", "JSON\n")):
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    candidate = text[start:] if start != -1 else text
    try:
        return json.loads(_repair_truncated_json(candidate))
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse critic JSON: {e}\n--- raw output ---\n{text[:500]}") from e


def _coerce_scores(raw: dict | None, axes: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for axis in axes:
            v = raw.get(axis, 5)
            try:
                out[axis] = max(1, min(10, int(v)))
            except (TypeError, ValueError):
                out[axis] = 5
    else:
        out = {axis: 5 for axis in axes}
    return out


# --- specialist base ---------------------------------------------------------

class _SpecialistCritic:
    """Base class. Subclasses define NAME, AXES, SYSTEM."""

    NAME: str = ""
    AXES: list[str] = []
    SYSTEM: str = ""

    def __init__(self, client: OpenAI, model: str, max_tokens: int = 2000, temperature: float = 0.2):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _schema_block(self) -> str:
        score_lines = "\n    ".join(f'"{axis}": int 1-10,' for axis in self.AXES).rstrip(",")
        return (
            "{\n"
            '  "scores": {\n'
            f"    {score_lines}\n"
            "  },\n"
            '  "strengths": [string, ...],\n'
            '  "weaknesses": [string, ...],\n'
            '  "suggestions": [string, ...]\n'
            "}"
        )

    def critique(self, static_context: str, job_offering: str, draft: str) -> SpecialistVerdict:
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "<draft_to_critique>\n"
            f"{draft.strip()}\n"
            "</draft_to_critique>\n\n"
            f"Score the draft on YOUR axes only ({', '.join(self.AXES)}). "
            f"Return ONLY this JSON object:\n{self._schema_block()}"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"{static_context}\n\n---\n\n{task}"},
            ],
            **chat_extra(),
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            data = _extract_json(raw)
        except ValueError:
            data = {}
        return SpecialistVerdict(
            name=self.NAME,
            scores=_coerce_scores(data.get("scores"), self.AXES),
            strengths=list(data.get("strengths", []) or []),
            weaknesses=list(data.get("weaknesses", []) or []),
            suggestions=list(data.get("suggestions", []) or []),
        )


# --- specialists -------------------------------------------------------------

class StyleCritic(_SpecialistCritic):
    NAME = "style"
    AXES = ["style_fit", "language", "conversationality"]
    SYSTEM = """You judge ONLY whether a motivation letter draft matches the example letters' voice and reads conversationally.

Score these axes 1-10 (be stingy, default 6 or 7):
  - style_fit         : tone, register, length, vocabulary match the example letter(s) in the static context.
  - language          : idiomatic, correct, no awkward phrasing, in the job posting's language.
  - conversationality : reads like a person talking, not a press release. No marketing tone. No metaphors like "un pont entre", "à la croisée de", "au cœur de" (any such metaphor caps this axis at 6). Sycophancy ("thrilled to apply", "passionate about your mission") caps both style_fit AND conversationality at 6.

The example letters in the static context are the ground truth. If the draft sounds noticeably more formal, more press-release, more verbose, or more flattering than them, you score it down.

You do NOT judge: factual accuracy, technical depth, AI-tell punctuation patterns, company recap. Other specialists handle those.

Be specific in weaknesses. Quote offending phrases verbatim. For each weakness, give a one-line concrete rewrite. Strengths only if genuinely true."""


class AntiAICritic(_SpecialistCritic):
    NAME = "anti_ai"
    AXES = ["no_ai_tells", "no_company_recap"]
    SYSTEM = """You hunt chatbot-writing patterns, formulaic structure, and company-recap sentences. Two axes only. Be paranoid: AI tells are often stylistic (tone, phrasing, rhythm), not factual; your job is to surface precise, actionable edits that make the draft read human.

Return ONLY the JSON object schema requested by the caller.

Scoring: 1-10 integers (be stingy; default ~5–6 when unsure). Apply the explicit caps below when micro- or macro-patterns are present.

Axes:
 - no_ai_tells: micro- and macro-level stylistic/structural signals that make text sound like an LLM or template.
 - no_company_recap: any sentence that explains the company's/program's/product's work back to the reader.

Micro-patterns (flag and cap scores as noted):
    - Em-/en-dashes used as clause separators ("—", "–") or excessive hyphen glue: ANY occurrence caps at 6.
    - Comma-fenced appositions: "X, [descriptor], Y" (e.g., "Mon profil, axé sur Y, me permet..."): ANY occurrence caps at 6.
    - Gratuitous parentheticals / long side-comments: penalize when not matching example usage.
    - LLM-jargon & corporate buzzwords in non-technical sentences (examples: leverage, robust, synergize, endeavor, passionate about, thrilled, deeply, foster, holistic, cutting-edge, state-of-the-art, seamlessly, comprehensive, strive, paradigm, ecosystem, valeur ajoutée, stratégie data, industrialiser): ANY occurrence caps at 7.
    - Generic scaffold transitions and boilerplate connectors (examples: "De plus", "En outre", "Par ailleurs", "Furthermore", "Moreover", "Additionally", "In addition", "Notamment", "Au-delà de la modélisation"): ANY occurrence caps at 7.
    - Overcomplicated vocabulary or nominalizations in otherwise non-technical sentences: penalize for substituting jargon for concreteness.
    - Excessive passive constructions and indirect speech (impersonal tone): penalize; prefer active first-person statements.
    - Mechanical transitions and subordinate-clause chaining (many subordinate clauses in a single sentence): penalize for reducing spontaneity.
    - Repetitive sentence openers and uniform sentence length (many sentences in the same 18–24 word band): penalize as an AI-tell.

Macro-patterns (read the whole draft; name the pattern and suggest a structural fix):
    - Square/templated paragraph shape where every paragraph mirrors the same structure (one idea → one example → one takeaway). If every paragraph matches the same shape, cap `no_ai_tells` at 6; suggest merging/splitting or adding an anecdote to vary rhythm.
    - Repetitive paragraph-level loops (observation → example → lesson) across the letter: cap at 6; suggest consolidation or a bridging reflection.
    - Suspiciously uniform paragraph length or pacing: cap at 6; suggest inserting at least one short sentence and one denser paragraph.
    - Formulaic flow driven by connectors rather than content-led transitions: identify repeated connectors and recommend replacing with content-led links.

Humanizing fixes to recommend (one-line suggestions):
    - Prefer active voice and concrete verbs; split long sentences into two.
    - Vary sentence length: insert at least one short sentence (<8 words) and one long sentence (>25 words) where natural.
    - Replace jargon with specific actions or short examples (quote offending phrase and provide exact one-line replacement).
    - Add a brief personal detail or single-sentence anecdote tied to the CV to break formality (if consistent with examples).
    - Use simpler, idea-driven transitions (e.g., "When I saw..., I...") rather than scaffold phrases.
    - Vary paragraph shapes: merge para N+M into a denser paragraph or split a long paragraph to create contrast.

no_company_recap:
    - Penalize any sentence that explains the company's work back to the reader (e.g., "Votre entreprise, leader in X," or "Le programme X, qui vise à..."). ANY such sentence caps at 6. For each flagged sentence, provide a one-line rewrite that either (a) removes the recap and states what the candidate will do, or (b) replaces the recap with a concrete, job-specific observation grounded in the posting.

What to return for each weakness:
    - Quote the offending phrase (or name the macro-pattern verbatim).
    - Give a one-line concrete rewrite (exact replacement text or a short structural instruction: "merge para 2+3 into one denser paragraph").
    - For macro-patterns, name the pattern and give a short structural fix.

What you DO NOT judge:
    - Style fit to the example letters, factual accuracy, idea quality, or conversationality (other critics handle those).

Tone for your analysis:
    - Precise, surgical, and actionable. Minimal hand-waving. Prioritize the single highest-impact fix per weakness.

Scoring guidance:
    - Be stingy on `no_ai_tells`; default to 5–6 if unsure.
    - Apply the caps above when exact micro/macro patterns appear.
    - Use integer scores 1–10 only.

If the draft is clean on these axes, return high scores and list one or two minor polish suggestions.
"""


class AuthenticityCritic(_SpecialistCritic):
    NAME = "authenticity"
    AXES = ["authenticity", "job_relevance"]
    SYSTEM = """You verify two things:
  - authenticity  : every claim in the draft is traceable to the CV in the static context. Fabricated employers, projects, certifications, or skills cap this axis at 4.
  - job_relevance : the letter speaks to the job's actual requirements (not generic). It picks 1-3 skills/experiences relevant to THIS posting, not a CV recap.

Score 1-10 (be stingy, default 6 or 7).

You do NOT judge: style, AI-tells, conversationality, idea quality.

For authenticity weaknesses, quote the unsupported claim and say which CV section it should match (or that no CV item supports it). For job_relevance weaknesses, point at which posting requirement is unaddressed or which CV item is irrelevant filler."""


class SubstanceCritic(_SpecialistCritic):
    NAME = "substance"
    AXES = ["thinking"]
    SYSTEM = """You judge ONE axis: does the letter engage with the job's actual problems and propose concrete approaches the candidate would take?

Score thinking 1-10 (be stingy, default 5 or 6):
  - 9-10: each major paragraph identifies a specific challenge from the posting and sketches a concrete approach grounded in one CV fact.
  - 7-8 : engages with the role's challenges but stays mostly at the level of restating skills.
  - 5-6 : a CV recap with vague nods to the job. CAP at 5 for pure CV recap.
  - 1-4 : generic, no engagement with the posting.

Reward: selective, well-placed experience supporting a concrete sketch.
Penalize: shopping list of credentials, "I have done X, Y, Z" without saying how it helps THIS role.

You do NOT judge: style, language, AI-tells, factual accuracy.

For weaknesses, quote the credential-dump or generic sentence verbatim and suggest a one-line concrete rewrite that proposes how the candidate would tackle a posting-specific problem."""


# --- deterministic metrics critic -------------------------------------------

class MetricsCritic:
    """Deterministic, no-LLM critic that scores a draft against quantitative
    style metrics computed from the user's example letters. These are the
    numbers AI detectors actually look at: sentence-length variance, lexical
    diversity, n-gram overlap with user vocabulary, punctuation distribution,
    and (optionally) echo-perplexity under the same model."""

    NAME = "metrics"
    AXES = METRIC_AXES

    def __init__(
        self,
        baseline: dict[str, dict[str, float]],
        examples_corpus: str = "",
        client: OpenAI | None = None,
        model: str = "",
        baseline_perplexity: float | None = None,
    ):
        self.baseline = baseline
        self.examples_corpus = examples_corpus
        self.client = client
        self.model = model
        self.baseline_perplexity = baseline_perplexity

    def critique(self, static_context: str, job_offering: str, draft: str) -> SpecialistVerdict:
        m = compute_metrics(draft, self.examples_corpus)
        scores: dict[str, int] = {}
        weaknesses: list[str] = []
        suggestions: list[str] = []

        # 1. burstiness - sentence length variance
        bv = m["sentence_length_variance"]
        target = self.baseline.get("sentence_length_variance")
        scores["burstiness_match"] = score_against_baseline(bv, target)
        if target and scores["burstiness_match"] < 7:
            weaknesses.append(
                f"sentence-length variance {bv:.1f} vs example baseline mean {target['mean']:.1f}; "
                f"sentences are too uniform (LLM tell)"
            )
            suggestions.append(
                "vary sentence length: insert at least one short sentence under 8 words and one over 25"
            )

        # 2. lexical diversity
        ld = m["lexical_diversity"]
        target = self.baseline.get("lexical_diversity")
        scores["lexical_diversity_match"] = score_against_baseline(ld, target)
        if target and scores["lexical_diversity_match"] < 7:
            direction = "too repetitive" if ld < target["mean"] else "too varied (over-thesaurus)"
            weaknesses.append(f"lexical diversity {ld:.2f} vs baseline {target['mean']:.2f}, {direction}")
            suggestions.append(
                "match the example letters' vocabulary range; reuse domain words instead of synonyms"
            )

        # 3. punctuation - composite of comma rate, em-dashes, parens
        cr = m["comma_rate"]
        cr_target = self.baseline.get("comma_rate")
        comma_score = score_against_baseline(cr, cr_target)
        em_count = int(m["em_dash_count"])
        em_penalty = max(1, 10 - em_count * 4)  # any em-dash is heavily penalized
        # parens are fine in moderation; only penalize gross over/underuse vs baseline
        pn = m["paren_count"]
        pn_target = self.baseline.get("paren_count")
        paren_score = score_against_baseline(pn, pn_target)
        scores["punctuation_match"] = min(comma_score, em_penalty, paren_score)
        if em_count > 0:
            weaknesses.append(f"contains {em_count} em-dash/en-dash separator(s) (AI tell); replace with commas or full stops")
        if cr_target and comma_score < 7:
            direction = "over-commaed" if cr > cr_target["mean"] else "under-commaed"
            weaknesses.append(f"comma rate {cr:.2f} per sentence vs baseline {cr_target['mean']:.2f}, {direction}")

        # 4. n-gram authenticity - bigram overlap with user's example corpus
        if self.examples_corpus:
            bigram_pct = m.get("bigram_overlap", 0.0) * 100
            scores["ngram_authenticity"] = score_ngram_overlap(m.get("bigram_overlap", 0.0))
            if scores["ngram_authenticity"] < 6:
                weaknesses.append(
                    f"only {bigram_pct:.1f}% of bigrams overlap with the user's example letters; "
                    "vocabulary feels generic"
                )
                suggestions.append("reuse phrases and word choices from the example letters where natural")
        else:
            scores["ngram_authenticity"] = 5

        # 5. predictability - echo-perplexity under the same model
        if self.client and self.model and self.baseline_perplexity is not None:
            ppl = echo_perplexity(self.client, self.model, draft)
            scores["predictability"] = score_predictability(ppl, self.baseline_perplexity)
            if ppl is not None and scores["predictability"] < 6:
                weaknesses.append(
                    f"echo-perplexity {ppl:.2f} below human-target baseline {self.baseline_perplexity:.2f}; "
                    "the model finds this draft too predictable (AI-detector signal)"
                )
                suggestions.append(
                    "introduce more unexpected word choices and sentence pivots; the example letters do this"
                )
        else:
            scores["predictability"] = 5

        return SpecialistVerdict(
            name=self.NAME,
            scores=scores,
            strengths=[],
            weaknesses=weaknesses,
            suggestions=suggestions,
        )


# --- panel -------------------------------------------------------------------

class CriticPanel:
    """Runs all specialists in parallel and merges into a single Critique.

    The deterministic MetricsCritic is added when a baseline (computed from
    the user's example letters) is provided. It runs alongside the LLM
    critics; per-axis ship thresholds in GATING_THRESHOLDS apply different
    bars to LLM axes (strict, 9) vs metric axes (lenient, 6)."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        metrics_baseline: dict[str, dict[str, float]] | None = None,
        examples_corpus: str = "",
        baseline_perplexity: float | None = None,
    ):
        self.client = client
        self.model = model
        self.llm_critics: list[_SpecialistCritic] = [
            StyleCritic(client, model),
            AntiAICritic(client, model),
            AuthenticityCritic(client, model),
            SubstanceCritic(client, model),
        ]
        self.metrics_critic: MetricsCritic | None = None
        if metrics_baseline:
            self.metrics_critic = MetricsCritic(
                baseline=metrics_baseline,
                examples_corpus=examples_corpus,
                client=client,
                model=model,
                baseline_perplexity=baseline_perplexity,
            )

    def critique(self, static_context: str, job_offering: str, draft: str) -> Critique:
        # LLM critics in parallel.
        with ThreadPoolExecutor(max_workers=len(self.llm_critics)) as pool:
            sub_verdicts: list[SpecialistVerdict] = list(
                pool.map(lambda c: c.critique(static_context, job_offering, draft), self.llm_critics)
            )

        # Deterministic metrics critic (synchronous, fast - no parallelization needed).
        if self.metrics_critic is not None:
            sub_verdicts.append(self.metrics_critic.critique(static_context, job_offering, draft))

        merged_scores: dict[str, int] = {}
        all_strengths: list[str] = []
        all_weaknesses: list[str] = []
        all_suggestions: list[str] = []
        by_critic: dict[str, SpecialistVerdict] = {}

        for sv in sub_verdicts:
            merged_scores.update(sv.scores)
            all_strengths.extend(f"[{sv.name}] {s}" for s in sv.strengths)
            all_weaknesses.extend(f"[{sv.name}] {w}" for w in sv.weaknesses)
            all_suggestions.extend(f"[{sv.name}] {s}" for s in sv.suggestions)
            by_critic[sv.name] = sv

        # Default any missing axis to 5 so a panel failure can't accidentally ship.
        for axis in ALL_AXES:
            merged_scores.setdefault(axis, 5)

        # Per-axis gating: LLM axes need >= 9, metric axes need >= 6.
        verdict = (
            "ship"
            if all(merged_scores[a] >= GATING_THRESHOLDS.get(a, SHIP_PER_AXIS_THRESHOLD) for a in ALL_AXES)
            else "revise"
        )

        return Critique(
            scores=merged_scores,
            verdict=verdict,
            strengths=all_strengths,
            weaknesses=all_weaknesses,
            suggestions=all_suggestions,
            by_critic=by_critic,
        )


# Backward-compat alias: the orchestrator imports CriticAgent.
CriticAgent = CriticPanel
