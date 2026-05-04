"""Crawl4AI's full LLMExtractionStrategy, schema-driven and chunked.

Unlike a single messages.create call against pruned markdown, this strategy:
- splits the page into overlapping token-bounded chunks,
- runs the LLM per chunk asking it to fill a JobPosting JSON schema,
- merges per-chunk extractions into one structured object.

That sidesteps two failure modes we hit with the standard approach: empty
responses on noisy pages, and the model fixating on an "offer no longer
available" banner instead of the posting body."""

import os

from crawl4ai import LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from llm import default_model

JOB_POSTING_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The job title."},
        "company": {"type": "string", "description": "The hiring organization name."},
        "location": {"type": "string", "description": "City, region, country."},
        "contract_type": {
            "type": "string",
            "description": "e.g. Stage, CDI, CDD, Internship, Full-time, Part-time.",
        },
        "duration": {"type": "string", "description": "e.g. '6 mois', '12 months'."},
        "salary": {"type": "string"},
        "remote_policy": {
            "type": "string",
            "description": "e.g. 'occasional telework', 'hybrid', 'fully remote'.",
        },
        "description": {
            "type": "string",
            "description": "Free-text overview / context paragraphs from the posting.",
        },
        "missions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet list of responsibilities / what the candidate will do.",
        },
        "requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet list of required profile / skills.",
        },
        "benefits": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet list of perks.",
        },
        "language": {
            "type": "string",
            "description": "BCP-47 language code of the posting itself: 'fr', 'en', 'es'.",
        },
    },
    "required": ["title"],
}

INSTRUCTION = """Extract the job posting from this page chunk into the JSON schema.

Rules:
- Use the exact wording from the page. Do not paraphrase, translate, or summarize.
- Preserve the original posting language in every string field.
- IGNORE banners saying the offer is no longer available, expired, or closed - the posting body itself is on the page; extract it.
- IGNORE site navigation, related-job lists, footer, cookie banners, "apply" buttons, ads.
- If the chunk contains no relevant posting content, return an object with empty strings / empty arrays."""


def make_strategy() -> LLMExtractionStrategy:
    # Single source of truth for the model name - same env var the rest of the
    # app uses, so creative-agent / gemma4 / whatever you set in .env applies here too.
    model = default_model()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Strip any trailing /v1 because LiteLLM's ollama provider expects the bare host.
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]

    llm_config = LLMConfig(
        provider=f"ollama/{model}",
        api_token="ollama",
        base_url=base_url,
    )

    return LLMExtractionStrategy(
        llm_config=llm_config,
        schema=JOB_POSTING_SCHEMA,
        extraction_type="schema",
        instruction=INSTRUCTION,
        chunk_token_threshold=1400,
        overlap_rate=0.1,
        apply_chunking=True,
        input_format="markdown",
        extra_args={
            "temperature": 0.0,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "16384")),
        },
    )
