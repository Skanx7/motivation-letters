# cover_letters

A local, multi-agent CLI that drafts a motivation letter for a job posting, then iterates against a panel of specialist critics until the letter reads like a human wrote it. Runs entirely on your machine via [Ollama](https://ollama.com); no API keys, no cloud calls.

Given:
- Your CV (`cv.pdf`)
- One or more example motivation letters that capture your voice (`motivation_examples.yaml`)
- A target job posting (URL or pasted text)

It produces a tailored letter in the same language as the posting, grounded in your CV, in your voice, and engineered to look as little like a chatbot wrote it as possible.

## How it works

Six agents wired together by an orchestrator:

1. **`JobRetrieverAgent`** — given a URL, fetches and extracts the posting via a deterministic three-layer pipeline:
   1. **JSON-LD** parsing of `schema.org/JobPosting` blocks (zero LLM cost; works on Welcome to the Jungle, LinkedIn, Indeed, Greenhouse, Lever, Workable, Workday, etc.)
   2. **Crawl4AI's `LLMExtractionStrategy`** — schema-driven, chunked LLM extraction when JSON-LD is absent
   3. **Pruned markdown** — Crawl4AI's `PruningContentFilter` output as a final fallback

   If you paste text instead of a URL, this is skipped.

2. **`IdeatorAgent`** — reads the posting CV-blind and brainstorms 3 to 5 concrete technical approaches a thoughtful applicant might take. Returns `(problem, idea, why_useful)` triples. Free of the gravitational pull of the candidate's existing skills, so the writer gets strategic options rather than recycled CV phrasing.

3. **`WriterAgent`** runs in two phases:
   - **Phase 1 (beam)**: generates K candidate style drafts in parallel at temperature 0.9, each focused only on matching the example letters' voice and grounding in the CV. The deterministic `MetricsCritic` ranks them and the best is picked.
   - **Phase 2**: weaves 1 to 2 ideas from the ideator into the chosen draft, only those the candidate can genuinely claim from the CV, while preserving voice.

4. **`CriticPanel`** — five specialists run in parallel, each scoring its own narrow set of axes:

   | Specialist | Axes | What it checks |
   |---|---|---|
   | `StyleCritic` | `style_fit`, `language`, `conversationality` | Tone match with the example letters; idiomatic; not press-release |
   | `AntiAICritic` | `no_ai_tells`, `no_company_recap` | Em-dashes, comma-fenced appositions, LLM vocab, generic transitions, square paragraph structure, explaining the company's own work back to them |
   | `AuthenticityCritic` | `authenticity`, `job_relevance` | Every claim traces to the CV; speaks to THIS posting |
   | `SubstanceCritic` | `thinking` | Engages with the role's actual problems vs being a CV recap |
   | `MetricsCritic` (deterministic, no LLM) | `burstiness_match`, `lexical_diversity_match`, `punctuation_match`, `ngram_authenticity`, `predictability` | Quantitative match against the user's example letters: sentence-length variance, type/token ratio, n-gram overlap, comma rate, em-dashes, optional echo-perplexity |

   Per-axis ship gating: LLM axes need ≥ 9, metric axes ≥ 6. Even one axis below its bar forces "revise". The "best" draft across iterations is tracked by the average across all axes (so improvements on most axes count even if one weak axis lingers).

5. **Revise loop** — the writer addresses the bottleneck specialist's complaints first, sweeps the rest. Best-of-N fallback so a late revision can't damage a previously-good draft.

6. **`StylizerAgent`** — one-shot finishing pass that adds 1 to 3 natural fillers (`bref`, `je pense`, `j'aimerais`, `du coup`, `honnêtement`) the example letters use, smooths any residual machine phrasing, and varies paragraph length. No critique loop; it's the final touch.

### Pipeline

```
STEP 0    Load CV + examples; compute deterministic style baseline + perplexity baseline
STEP 1    Retriever               URL or stdin -> clean job text
STEP 2    Ideator                 K technical ideas, CV-blind
STEP 3a   Writer phase 1 (beam)   K parallel drafts, MetricsCritic picks best
STEP 3b   Writer phase 2          weave 1-2 fitting ideas into the draft
STEP 4.N  Critic panel            4 LLM specialists + 1 metric specialist
STEP 5.N  Writer revise           bottleneck-first
STEP 6    Stylizer                final humanizing pass on the best-of-N draft
FINAL LETTER
```

## Requirements

- **Python 3.10+** (uses `str | None` syntax)
- **Ollama** — runs the local LLM
- A GPU with ≥ 14 GB VRAM is recommended for the 12B-class model used by default. Smaller / larger variants are documented in the `Modelfile`.

## Installation

### 1. Clone and create a virtual environment

```sh
git clone <your-fork-url> motivation_letters
cd motivation_letters
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 2. Install Python dependencies

```sh
pip install -r requirements.txt
```

### 3. Install Crawl4AI's Playwright browsers (one-time)

```sh
crawl4ai-setup
```

If that fails on Windows, the manual fallback is:

```sh
python -m playwright install chromium
```

### 4. Install Ollama

- **Windows / macOS** — download the installer from https://ollama.com/download
- **Linux** — `curl -fsSL https://ollama.com/install.sh | sh`

Confirm it's running:

```sh
ollama --version
ollama list
```

### 5. Pull a base model and build the project's tuned model

The agents read `OLLAMA_MODEL` from your `.env`. The default name is `creative-agent`, built from the `Modelfile` in this repo:

```sh
# Pull whatever base model the Modelfile points at
ollama pull gemma4:12b

# Build the tuned model with the wider context window
ollama create creative-agent -f Modelfile
```

The `Modelfile` sets `num_ctx=32768` (or 65536 if you bumped it) so the agents have enough headroom for crawled markdown + CV + examples + ideator output without silent truncation. Edit the `FROM` line if your base model differs.

### 6. Configure your environment

```sh
cp .env.example .env
```

`.env` keys:

| Key | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `creative-agent` | Model name (must match what you built in step 5) |
| `OLLAMA_NUM_CTX` | `16384` | Override Ollama's default 2048-token window — required to stop silent truncation on long inputs |

### 7. Drop your own data in

- Replace `cv.pdf` with your own CV.
- Edit `motivation_examples.yaml` to contain `(job_offering, motivation)` pairs that demonstrate your writing voice.
- **More examples = better detection-resistance.** With one example, the deterministic baseline has to fall back to a generous std. With 3 to 5 examples, sentence-length variance, lexical diversity, and n-gram overlap targets become much sharper, and the writer's beam phase has more signal to rank against.

## Usage

### From a job posting URL

```sh
python main.py https://www.welcometothejungle.com/fr/companies/<company>/jobs/<slug>
```

### Interactive paste

```sh
python main.py
# Paste the URL (single Enter to submit), OR paste the full job text and end with a line containing only END
```

The final letter is written to **stdout**; per-step diagnostics go to **stderr**. So:

```sh
python main.py <url> > letter.txt
```

### What you'll see in the trace

The orchestrator prints, in order:
- the static-context baseline numbers (sentence length variance, comma rate, etc., computed from your example letters)
- the retrieved + extracted job text
- each ideator triple
- per-candidate metric scores from the beam phase 1
- the chosen style draft, then the enriched draft after phase 2
- per-iteration: each specialist's per-axis scores, weaknesses, suggestions, the bottleneck axis, the new-best announcement
- the final humanized letter

### Debug artifacts

After each run, the retriever dumps to the project root:

- `.crawl_last.html` — raw HTML from Crawl4AI
- `.crawl_last_raw.md` — unfiltered markdown
- `.crawl_last.md` — pruned markdown
- `.crawl_last_llm.json` — raw output from `LLMExtractionStrategy` (chunked JSON)
- `.crawl_last_extracted.txt` — final clean job text used by the writer

Add them to `.gitignore` if you start versioning the repo.

## Project structure

```
motivation_letters/
├── cv.pdf                          (your CV, replace with your own)
├── motivation_examples.yaml        (your voice, edit with your own examples)
├── Modelfile                       (Ollama Modelfile for creative-agent)
├── requirements.txt
├── .env.example
├── main.py                         (CLI entry point, ~14 lines)
├── llm.py                          (Ollama client factory + shared options like num_ctx)
├── context.py                      (CV + examples loader, builds cached static LLM context)
├── probe.py                        (diagnostic script: tests the model end-to-end)
├── agents/
│   ├── orchestrator.py             (conducts the pipeline; owns the agents and the loop)
│   ├── retriever.py                (CLI/IO; delegates to extraction/)
│   ├── ideator.py                  (CV-blind technical brainstormer)
│   ├── writer.py                   (two-phase: draft_style + incorporate_ideas + revise)
│   ├── stylizer.py                 (final humanizing pass)
│   ├── critic.py                   (4 LLM specialists + MetricsCritic + CriticPanel aggregator)
│   ├── style_metrics.py            (deterministic burstiness/diversity/n-gram/punctuation)
│   └── perplexity.py               (echo-and-score via Ollama logprobs)
└── extraction/
    ├── crawler.py                  (Crawl4AI wrapper)
    ├── jsonld.py                   (Layer 1: schema.org JobPosting parser)
    ├── llm_extraction.py           (Layer 2: Crawl4AI LLMExtractionStrategy)
    ├── pipeline.py                 (orchestrates all three extraction layers)
    └── models.py                   (JobPosting dataclass)
```

## Tuning for naturalness and against AI detectors

The bigger the gap between LLM-generated text and human writing on quantitative dimensions like burstiness, lexical diversity, and n-gram authenticity, the easier any AI detector spots it. The pipeline scores every draft against your own example letters on those exact dimensions and feeds the gaps into the writer's revision step.

What helps most, in order:

1. **Add more example letters.** Three to five real letters (any topic — internships, personal projects, even Reddit posts in the right tone) drop detection rates dramatically. The metrics baseline gets sharper, the n-gram overlap target becomes meaningful, and the model has more voice signal to imitate. Free, no code changes.
2. **One round of human editing.** Retyping a single paragraph in your own words destroys most detection signals. Detectors are calibrated on full machine output; partial human authorship is their failure mode.
3. **Let phase 1 beam wider.** `Orchestrator(beam_size=10)` produces 10 parallel drafts at temp 0.9 and lets `MetricsCritic` pick the best. Higher temp + more candidates = better outliers in your style. Costs more LLM calls.
4. **Compute the perplexity baseline.** Default-on. Disable with `Orchestrator(compute_perplexity_baseline=False)` only if startup time matters more than the predictability axis.
5. **Different stylizer model.** A different Ollama model for the stylizer than the writer breaks any single-fingerprint signal. See "Customizing" below.

## Customizing

The orchestrator is a class with constructor params, not a script with magic constants:

```python
from agents import Orchestrator

Orchestrator(
    cv_path="cv.pdf",
    examples_path="motivation_examples.yaml",
    max_iters=5,                       # critic-revise rounds before stopping
    ship_threshold=8,                  # ship gate (per-axis thresholds in critic.py override this for finer control)
    beam_size=5,                       # parallel candidate drafts in phase 1
    compute_perplexity_baseline=True,  # one extra echo per example at startup; off = neutral predictability
).run(url=...)
```

Other knobs:

- **Voice / language emphasis** — edit `motivation_examples.yaml`. The writer and stylizer imitate whatever's there.
- **Per-agent system prompts** — top of `agents/writer.py`, `agents/critic.py`, `agents/ideator.py`, `agents/stylizer.py`.
- **Per-axis ship thresholds** — `GATING_THRESHOLDS` in [agents/critic.py](agents/critic.py). Defaults: 9 for the 8 LLM axes, 6 for the 5 metric axes.
- **Extraction priority** — reorder layers in `extraction/pipeline.py:extract_from_url`.
- **Stylizer's filler set** — `STYLIZER_SYSTEM` in [agents/stylizer.py](agents/stylizer.py). Add or remove the interjection words the model is allowed to inject.

## Troubleshooting

**`OllamaException - model 'X' not found`**
Either `OLLAMA_MODEL` in your `.env` doesn't match what's in `ollama list`, or you skipped step 5. Run `ollama list` and either rename your `.env` value or rebuild with `ollama create creative-agent -f Modelfile`.

**Agents return empty / hallucinated content**
Almost always Ollama's 2048-token default context window silently truncating. Confirm `OLLAMA_NUM_CTX` is set in `.env` AND that your `Modelfile` sets `PARAMETER num_ctx 32768` (or higher). Both layers must agree — Modelfile sets the upper bound, env var sets the request value.

**`crawl4ai-setup` fails on Windows**
Run `python -m playwright install chromium` instead. Needs ~200 MB and a Visual C++ runtime.

**Crawl returns mostly nav/footer / pruning gives bad fragments**
Use the `.crawl_last.html` dump to confirm the page actually loaded. JS-heavy boards (some Workday tenants, LinkedIn behind auth) can return near-empty HTML to Playwright. The JSON-LD layer in `extraction/jsonld.py` is the most reliable path; if it's not finding a `JobPosting` block on a board you use often, the page may not embed one and you're better off pasting the text directly.

**Critic JSON parse fails**
The pipeline already self-repairs truncated JSON, but if you see this often: bump `num_predict` in the `Modelfile` to 4096+ and `OLLAMA_NUM_CTX` to 32768+.

**Predictability axis stuck at 5**
That means echo-perplexity returned `None` — your Ollama version or model doesn't expose `logprobs` on the OpenAI-compat endpoint. The pipeline falls back to a neutral score (5) and continues; the other 12 axes still work. Update Ollama or pick a model whose adapter exposes logprobs.

**Beam phase always picks candidate 1**
Either `temperature=0.9` isn't doing enough on this model, or your baseline doesn't differentiate enough. Try raising `Orchestrator(beam_size=10)` and adding more example letters.

**Run the diagnostic probe**

```sh
python probe.py
```

Hits the model with progressively bigger inputs and tells you where things break (model down, JSON mode broken, content too long, logprobs missing, etc.).
