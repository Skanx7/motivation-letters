import os
import sys

from dotenv import load_dotenv

from agents import CriticAgent, JobRetrieverAgent, WriterAgent
from context import build_static_context, load_cv, load_examples
from llm import default_model, make_client

MAX_ITERS = 4
SHIP_THRESHOLD = 8


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}", file=sys.stderr)


def _step(label: str) -> None:
    print(f"\n--- {label} ---", file=sys.stderr)


def _print_critique(c) -> None:
    print(f"score:   {c.score}/10", file=sys.stderr)
    print(f"verdict: {c.verdict}", file=sys.stderr)
    for field, items in (("strengths", c.strengths), ("weaknesses", c.weaknesses), ("suggestions", c.suggestions)):
        print(f"{field}:", file=sys.stderr)
        if not items:
            print("  (none)", file=sys.stderr)
        for it in items:
            print(f"  - {it}", file=sys.stderr)


def main() -> None:
    load_dotenv()

    url = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith(("http://", "https://")) else None

    _section("STEP 0  Loading static context")
    print("Reading cv.pdf and motivation_examples.yaml ...", file=sys.stderr)
    cv = load_cv("cv.pdf")
    examples = load_examples("motivation_examples.yaml")
    static_ctx = build_static_context(cv, examples)
    print(f"CV: {len(cv)} chars | examples: {len(examples)} | combined context: {len(static_ctx)} chars", file=sys.stderr)

    client = make_client()
    model = default_model()
    print(f"LLM: {model} @ {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}", file=sys.stderr)

    retriever = JobRetrieverAgent(client, model=model)
    writer = WriterAgent(client, model=model)
    critic = CriticAgent(client, model=model)

    _section("STEP 1  Retriever agent  (URL or stdin -> clean job text)")
    if url:
        print(f"source: {url}", file=sys.stderr)
    else:
        print("source: stdin (no URL given)", file=sys.stderr)
    job = retriever.get(url)
    _step("Retrieved job posting")
    print(job, file=sys.stderr)

    _section("STEP 2  Writer agent  (initial draft)")
    print("Generating initial draft from CV + examples + job ...", file=sys.stderr)
    draft = writer.draft(static_ctx, job)
    _step("Draft v0")
    print(draft, file=sys.stderr)

    final_score = None
    final_iter = 0
    for i in range(1, MAX_ITERS + 1):
        _section(f"STEP 3.{i}  Critic agent  (review draft v{i - 1})")
        critique = critic.critique(static_ctx, job, draft)
        _print_critique(critique)

        final_score = critique.score
        final_iter = i

        if critique.verdict == "ship" or critique.score >= SHIP_THRESHOLD:
            print(f"\n>>> Critic accepted at score {critique.score}/10. Stopping loop.", file=sys.stderr)
            break

        if i == MAX_ITERS:
            print(f"\n>>> Max iterations ({MAX_ITERS}) reached without convergence.", file=sys.stderr)
            break

        _section(f"STEP 4.{i}  Writer agent  (revise -> draft v{i})")
        print("Revising in response to critic feedback ...", file=sys.stderr)
        draft = writer.revise(static_ctx, job, draft, critique.to_dict())
        _step(f"Draft v{i}")
        print(draft, file=sys.stderr)

    _section(f"FINAL LETTER  (score {final_score}/10 after {final_iter} critique round(s))")
    print(draft)


if __name__ == "__main__":
    main()
