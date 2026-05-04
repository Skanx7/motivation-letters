"""Three-layer extraction pipeline. ONE crawler, deterministic where possible.

Layer 1 - JSON-LD (`schema.org/JobPosting`): every modern board (Welcome to
the Jungle, LinkedIn, Indeed, Greenhouse, Lever, Workable, SmartRecruiters,
Workday, Welcome Kit) embeds machine-readable JobPosting structured data in
the page HTML. Parsing it gives title/company/location/contract and a clean
HTML description with no LLM. Handles ~80-90% of URLs.

Layer 2 - Crawl4AI's full LLMExtractionStrategy: schema-driven, chunked
extraction running inside the crawler. Used when JSON-LD is absent OR thin.
Handles long pages by splitting them and merging per-chunk results, which is
why it doesn't fail the way a single-shot messages.create call did on noisy
pruned markdown.

Layer 3 - pruned markdown directly: graceful degradation if both layers
above fail. Crawl4AI's PruningContentFilter output is small and focused; not
as clean as the structured layers but still deterministic."""

import json
from dataclasses import dataclass, field
from typing import Literal

from . import jsonld
from .crawler import CrawlOutput, crawl
from .llm_extraction import make_strategy
from .models import JobPosting

MIN_USEFUL_LEN = 200


@dataclass
class ExtractionResult:
    url: str
    crawl: CrawlOutput | None = None
    jsonld_posting: JobPosting | None = None
    llm_posting: JobPosting | None = None
    final_text: str = ""
    method: Literal["jsonld", "llm_extraction", "pruned_markdown", "failed"] = "failed"
    notes: list[str] = field(default_factory=list)


def _coerce_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(_coerce_str(x) for x in v if x).strip()
    return str(v).strip()


def _coerce_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [_coerce_str(x) for x in v if _coerce_str(x)]
    return [_coerce_str(v)] if _coerce_str(v) else []


def _merge_chunked_extractions(items: list[dict]) -> JobPosting:
    """LLMExtractionStrategy returns a list of per-chunk dicts. Merge them:
    take the first non-empty value for scalar fields, union the lists."""
    posting = JobPosting()
    scalar_fields = (
        "title", "company", "location", "contract_type", "duration",
        "salary", "remote_policy", "description", "language",
    )
    list_fields = ("missions", "requirements", "benefits")

    for item in items:
        if not isinstance(item, dict):
            continue
        for f in scalar_fields:
            if not getattr(posting, f):
                val = _coerce_str(item.get(f))
                if val:
                    setattr(posting, f, val)
        for f in list_fields:
            existing = getattr(posting, f)
            for v in _coerce_list(item.get(f)):
                if v not in existing:
                    existing.append(v)
    return posting


def _parse_llm_extracted(content: str) -> JobPosting | None:
    if not content:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    return _merge_chunked_extractions(data)


def extract_from_url(url: str) -> ExtractionResult:
    result = ExtractionResult(url=url)

    # Single crawl, with the LLM extraction strategy attached so layers 1 + 2
    # are produced from the same page fetch.
    strategy = make_strategy()
    out = crawl(url, extraction_strategy=strategy)
    result.crawl = out
    result.notes.append(
        f"crawl: html={len(out.raw_html)} chars, raw_md={len(out.raw_markdown)} chars, "
        f"fit_md={len(out.fit_markdown)} chars, llm_extracted={len(out.extracted_content)} chars"
    )

    # Layer 1: JSON-LD (deterministic, gold standard)
    jsonld_posting = jsonld.find_jobposting(out.raw_html)
    if jsonld_posting is not None:
        result.jsonld_posting = jsonld_posting
        if jsonld_posting.has_substantive_content():
            result.final_text = jsonld_posting.to_text()
            result.method = "jsonld"
            result.notes.append(
                f"jsonld: parsed JobPosting "
                f"(title={jsonld_posting.title!r}, description={len(jsonld_posting.description)} chars)"
            )
            return result
        result.notes.append("jsonld: found JobPosting block but description was thin; trying LLM extraction")
    else:
        result.notes.append("jsonld: no JobPosting block found in page HTML")

    # Layer 2: Crawl4AI's chunked LLMExtractionStrategy result
    llm_posting = _parse_llm_extracted(out.extracted_content)
    if llm_posting is not None:
        result.llm_posting = llm_posting
        if llm_posting.has_substantive_content():
            result.final_text = llm_posting.to_text()
            result.method = "llm_extraction"
            result.notes.append(
                f"llm_extraction: merged chunked output "
                f"(title={llm_posting.title!r}, missions={len(llm_posting.missions)}, "
                f"requirements={len(llm_posting.requirements)})"
            )
            return result
        result.notes.append("llm_extraction: merged output had no substantive content; falling through")
    else:
        result.notes.append("llm_extraction: returned no parseable JSON")

    # Layer 3: pruned markdown directly
    md = out.fit_markdown if len(out.fit_markdown) >= MIN_USEFUL_LEN else out.raw_markdown
    if len(md) >= MIN_USEFUL_LEN:
        result.final_text = md
        result.method = "pruned_markdown"
        result.notes.append(f"pruned_markdown: using {len(md)} chars as final text")
        return result

    result.notes.append(f"all layers failed (best md candidate was {len(md)} chars)")
    return result
