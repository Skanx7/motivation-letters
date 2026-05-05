import os
import sys

from openai import OpenAI

from context import build_static_context, load_cv, load_examples
from llm import default_model, make_client

from .critic import CriticPanel, Critique, MetricsCritic
from .ideator import Idea, IdeatorAgent
from .perplexity import baseline_perplexity_from_examples
from .retriever import JobRetrieverAgent
from .style_metrics import compute_baseline
from .stylizer import StylizerAgent
from .writer import WriterAgent


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}", file=sys.stderr)


def _step(label: str) -> None:
    print(f"\n--- {label} ---", file=sys.stderr)


def _print_critique(c: Critique) -> None:
    print("Critic panel verdict (every axis must clear ship threshold):", file=sys.stderr)
    width = max((len(a) for a in c.scores), default=0)
    bottleneck = c.bottleneck_axis
    for sv_name, sv in c.by_critic.items():
        print(f"  --- specialist: {sv_name} ---", file=sys.stderr)
        for axis, val in sv.scores.items():
            marker = "  <- bottleneck" if axis == bottleneck else ""
            print(f"    {axis.ljust(width)} : {val}/10{marker}", file=sys.stderr)
        for w in sv.weaknesses:
            print(f"    weakness   : {w}", file=sys.stderr)
        for s in sv.suggestions:
            print(f"    suggestion : {s}", file=sys.stderr)
    print(
        f"\nmin={c.min_score}/10  avg={c.avg_score:.1f}/10  verdict={c.verdict}  "
        f"bottleneck={bottleneck} (specialist: {c.bottleneck_critic})",
        file=sys.stderr,
    )


def _print_ideas(ideas: list[Idea]) -> None:
    if not ideas:
        print("No ideas produced. Writer will proceed without strategic input.", file=sys.stderr)
        return
    for i, idea in enumerate(ideas, 1):
        _step(f"Idea {i}")
        print(f"problem    : {idea.problem}", file=sys.stderr)
        print(f"approach   : {idea.idea}", file=sys.stderr)
        print(f"why_useful : {idea.why_useful}", file=sys.stderr)


class Orchestrator:
    """Conducts the agent pipeline: retriever -> ideator -> writer -> critic loop.

    Owns:
      - one shared LLM client and model name
      - the four agents (instantiated once, reused per run)
      - per-step logging to stderr
      - the critique/revise convergence loop
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        cv_path: str = "cv.pdf",
        examples_path: str = "motivation_examples.yaml",
        max_iters: int = 9,
        ship_threshold: int = 8,
        beam_size: int = 5,
        compute_perplexity_baseline: bool = True,
    ):
        self.client = client or make_client()
        self.model = model or default_model()
        self.cv_path = cv_path
        self.examples_path = examples_path
        self.max_iters = max_iters
        self.ship_threshold = ship_threshold
        self.beam_size = beam_size
        self.compute_perplexity_baseline = compute_perplexity_baseline

        # Filled in lazily by run() before any agents need them.
        self.metrics_baseline: dict[str, dict[str, float]] = {}
        self.examples_corpus: str = ""
        self.baseline_perplexity: float | None = None

        self.retriever = JobRetrieverAgent(self.client, model=self.model)
        self.ideator = IdeatorAgent(self.client, model=self.model)
        self.writer = WriterAgent(self.client, model=self.model)
        self.stylizer = StylizerAgent(self.client, model=self.model)
        # Critic + standalone metrics critic are constructed in _load_static_context
        # once the example-letter baseline is ready.
        self.critic: CriticPanel | None = None
        self.metrics_critic: MetricsCritic | None = None

    def run(self, url: str | None = None) -> str:
        """Run the full pipeline. Returns the final letter text."""
        ctx = self._load_static_context()
        job = self._retrieve_job(url)
        ideas = self._brainstorm(job)
        draft = self._initial_draft(ctx, job, ideas)
        draft, score, iters = self._critique_revise_loop(ctx, job, ideas, draft)
        draft = self._humanize(ctx, job, draft)
        _section(f"FINAL LETTER  (best avg score {score:.2f}/10 after {iters} critique round(s), humanized)")
        return draft

    # --- pipeline steps ------------------------------------------------------

    def _load_static_context(self) -> str:
        _section("STEP 0  Loading static context + computing style baseline")
        print("Reading CV and motivation_examples.yaml ...", file=sys.stderr)
        cv = load_cv(self.cv_path)
        examples = load_examples(self.examples_path)
        ctx = build_static_context(cv, examples)
        print(
            f"CV: {len(cv)} chars | examples: {len(examples)} | combined: {len(ctx)} chars",
            file=sys.stderr,
        )
        print(
            f"LLM: {self.model} @ {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}",
            file=sys.stderr,
        )

        # Compute deterministic style baseline from the example motivation letters.
        example_motivations = [ex["motivation"] for ex in examples if ex.get("motivation")]
        self.examples_corpus = "\n\n".join(example_motivations)
        self.metrics_baseline = compute_baseline(example_motivations)
        if self.metrics_baseline:
            blines = ", ".join(
                f"{k}={v['mean']:.2f}±{v['std']:.2f}"
                for k, v in sorted(self.metrics_baseline.items())
            )
            print(f"Style baseline from {len(example_motivations)} example letter(s): {blines}", file=sys.stderr)
        else:
            print("No example motivations available — MetricsCritic will be disabled.", file=sys.stderr)

        # Optionally compute the human-target perplexity baseline (one echo per example).
        if self.compute_perplexity_baseline and example_motivations:
            print("Computing echo-perplexity baseline from example letters (1 LLM call per example) ...", file=sys.stderr)
            self.baseline_perplexity = baseline_perplexity_from_examples(self.client, self.model, example_motivations)
            if self.baseline_perplexity is not None:
                print(f"Baseline perplexity: {self.baseline_perplexity:.3f}", file=sys.stderr)
            else:
                print("Echo-perplexity unavailable on this model/endpoint; predictability axis will be neutral.", file=sys.stderr)

        # Build the critic panel and a standalone metrics critic for beam ranking.
        self.critic = CriticPanel(
            self.client,
            model=self.model,
            metrics_baseline=self.metrics_baseline,
            examples_corpus=self.examples_corpus,
            baseline_perplexity=self.baseline_perplexity,
        )
        self.metrics_critic = (
            MetricsCritic(
                baseline=self.metrics_baseline,
                examples_corpus=self.examples_corpus,
                client=self.client,
                model=self.model,
                baseline_perplexity=self.baseline_perplexity,
            )
            if self.metrics_baseline
            else None
        )

        return ctx

    def _retrieve_job(self, url: str | None) -> str:
        _section("STEP 1  Retriever agent  (URL or stdin -> clean job text)")
        print(f"source: {url}" if url else "source: stdin (no URL given)", file=sys.stderr)
        job = self.retriever.get(url)
        _step("Retrieved job posting")
        print(job, file=sys.stderr)
        return job

    def _brainstorm(self, job: str) -> list[Idea]:
        _section("STEP 2  Ideator agent  (brainstorm approaches for the role)")
        print("Generating ideas from the posting alone, CV-blind ...", file=sys.stderr)
        ideas = self.ideator.brainstorm(job)
        _print_ideas(ideas)
        return ideas

    def _initial_draft(self, ctx: str, job: str, ideas: list[Idea]) -> str:
        _section(f"STEP 3a  Writer phase 1: BEAM of {self.beam_size} style drafts at temp 0.9")
        candidates = self.writer.draft_style_beam(ctx, job, k=self.beam_size, temperature=0.9)
        print(f"Generated {len(candidates)} candidate draft(s) in parallel.", file=sys.stderr)

        if not candidates:
            raise RuntimeError("beam writer produced zero drafts")

        if self.metrics_critic is not None:
            print("Ranking candidates by deterministic style metrics (no LLM) ...", file=sys.stderr)
            scored: list[tuple[float, int, str]] = []
            for i, cand in enumerate(candidates, 1):
                v = self.metrics_critic.critique(ctx, job, cand)
                avg = sum(v.scores.values()) / max(len(v.scores), 1)
                scored.append((avg, i, cand))
                axes_str = " ".join(f"{k.split('_')[0]}={s}" for k, s in v.scores.items())
                print(f"  candidate {i}: avg={avg:.2f}  [{axes_str}]", file=sys.stderr)
            scored.sort(key=lambda x: x[0], reverse=True)
            best_avg, best_idx, style_draft = scored[0]
            print(f">>> Selected candidate {best_idx} (metrics avg {best_avg:.2f}/10)", file=sys.stderr)
        else:
            style_draft = candidates[0]
            print(">>> No baseline available, using first candidate.", file=sys.stderr)

        _step("Draft v0a (best of beam, style only)")
        print(style_draft, file=sys.stderr)

        if not ideas:
            print("\nNo ideas to incorporate. Skipping phase 2.", file=sys.stderr)
            return style_draft

        _section("STEP 3b  Writer phase 2: incorporate ideator's ideas")
        print("Filtering ideas against CV + examples and weaving them in while preserving voice ...", file=sys.stderr)
        enriched = self.writer.incorporate_ideas(ctx, job, style_draft, ideas)
        _step("Draft v0b (style + ideas)")
        print(enriched, file=sys.stderr)
        return enriched

    def _critique_revise_loop(
        self, ctx: str, job: str, ideas: list[Idea], draft: str
    ) -> tuple[str, float, int]:
        assert self.critic is not None, "critic panel was not initialized; call _load_static_context first"
        """Iterate critique -> revise. Track the best-scoring draft (by AVERAGE
        across axes, not min) so improvements on most axes count even if one
        weak axis lingers. Note: ship gating still uses min via critique.verdict,
        but selection of the best-of-N draft uses avg.
        Returns (best_draft, best_avg, best_iter)."""
        best_draft = draft
        best_avg = -1.0
        best_min = -1
        best_iter = 0
        last_avg = -1.0
        last_min = -1
        last_iter = 0

        for i in range(1, self.max_iters + 1):
            _section(f"STEP 4.{i}  Critic panel  (review draft v{i - 1})")
            critique = self.critic.critique(ctx, job, draft)
            _print_critique(critique)
            last_avg = critique.avg_score
            last_min = critique.min_score
            last_iter = i

            if critique.avg_score > best_avg:
                best_avg = critique.avg_score
                best_min = critique.min_score
                best_draft = draft
                best_iter = i
                print(
                    f">>> New best draft: iter {i}, avg {best_avg:.2f}/10 (min {best_min}/10)",
                    file=sys.stderr,
                )

            if critique.verdict == "ship" or critique.score >= self.ship_threshold:
                print(
                    f"\n>>> Panel accepted (min-axis score {critique.score}/10). Stopping loop.",
                    file=sys.stderr,
                )
                break

            if i == self.max_iters:
                print(
                    f"\n>>> Max iterations ({self.max_iters}) reached without convergence.",
                    file=sys.stderr,
                )
                break

            _section(f"STEP 5.{i}  Writer agent  (revise -> draft v{i})")
            print(
                f"Revising. Bottleneck: '{critique.bottleneck_axis}' (specialist: {critique.bottleneck_critic}).",
                file=sys.stderr,
            )
            draft = self.writer.revise(ctx, job, draft, critique.to_dict(), ideas=ideas)
            _step(f"Draft v{i}")
            print(draft, file=sys.stderr)

        if best_iter != last_iter:
            print(
                f"\n>>> Best-of-iterations fallback: using draft from iter {best_iter} "
                f"(avg {best_avg:.2f}/10, min {best_min}/10) over last iter {last_iter} "
                f"(avg {last_avg:.2f}/10, min {last_min}/10).",
                file=sys.stderr,
            )
        return best_draft, best_avg, best_iter

    def _humanize(self, ctx: str, job: str, draft: str) -> str:
        _section("STEP 6  Stylizer agent  (final humanizing pass)")
        print(
            "Adding natural fillers and smoothing residual machine phrasing. One shot, no critique.",
            file=sys.stderr,
        )
        humanized = self.stylizer.humanize(ctx, job, draft)
        _step("Final humanized draft")
        print(humanized, file=sys.stderr)
        return humanized
