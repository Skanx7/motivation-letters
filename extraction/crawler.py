import asyncio
from dataclasses import dataclass
from typing import Optional

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import ExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


@dataclass
class CrawlOutput:
    url: str
    raw_html: str
    raw_markdown: str
    fit_markdown: str          # noise-pruned markdown
    extracted_content: str     # JSON string from the LLMExtractionStrategy, "" if none ran


async def _crawl(url: str, extraction_strategy: Optional[ExtractionStrategy]) -> CrawlOutput:
    md_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.48,
            threshold_type="dynamic",
            min_word_threshold=5,
        )
    )
    config_kwargs: dict = {"markdown_generator": md_gen}
    if extraction_strategy is not None:
        config_kwargs["extraction_strategy"] = extraction_strategy
    config = CrawlerRunConfig(**config_kwargs)

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url, config=config)
        if not getattr(result, "success", True):
            raise RuntimeError(f"crawl failed: {getattr(result, 'error_message', 'unknown error')}")
        html = getattr(result, "html", "") or getattr(result, "cleaned_html", "") or ""
        md = result.markdown
        if hasattr(md, "fit_markdown"):
            fit = md.fit_markdown or ""
            raw = md.raw_markdown or ""
        else:
            fit = md or ""
            raw = md or ""
        extracted = getattr(result, "extracted_content", "") or ""
        return CrawlOutput(
            url=url,
            raw_html=html,
            raw_markdown=raw,
            fit_markdown=fit,
            extracted_content=extracted,
        )


def crawl(url: str, extraction_strategy: Optional[ExtractionStrategy] = None) -> CrawlOutput:
    """Sync wrapper. If extraction_strategy is given, Crawl4AI runs it as part
    of the crawl pipeline and stores the result in CrawlOutput.extracted_content."""
    return asyncio.run(_crawl(url, extraction_strategy))
