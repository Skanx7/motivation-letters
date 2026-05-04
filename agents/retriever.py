import sys
from pathlib import Path

from openai import OpenAI

from extraction import ExtractionResult, extract_from_url

CRAWL_HTML_DUMP_PATH = Path(".crawl_last.html")
CRAWL_RAW_DUMP_PATH = Path(".crawl_last_raw.md")
CRAWL_FIT_DUMP_PATH = Path(".crawl_last.md")
CRAWL_LLM_DUMP_PATH = Path(".crawl_last_llm.json")
EXTRACT_DUMP_PATH = Path(".crawl_last_extracted.txt")

MIN_USEFUL_LEN = 200
PREVIEW_HEAD = 1500


class JobRetrieverAgent:
    """Owns CLI/IO concerns. Delegates the actual web -> clean text work to
    the extraction package."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def from_url(self, url: str) -> str:
        print(f"Crawling + extracting {url} ...", file=sys.stderr)
        result = extract_from_url(url)
        self._dump_all(result)
        self._print_diagnostic(result)

        if result.method == "failed" or len(result.final_text) < MIN_USEFUL_LEN:
            raise ValueError(
                f"All extraction layers failed for {url}. "
                f"See {CRAWL_HTML_DUMP_PATH.resolve()} and the other .crawl_last_* dumps."
            )

        print(
            f"Extraction succeeded via {result.method}: {len(result.final_text)} chars.",
            file=sys.stderr,
        )
        return result.final_text

    def from_stdin(self) -> str:
        print(
            "Paste the job URL (single Enter to submit), or paste the job text "
            "and finish with a line containing only END:",
            file=sys.stderr,
        )
        lines: list[str] = []
        try:
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                stripped = line.strip()
                if not lines and stripped.startswith(("http://", "https://")) and " " not in stripped:
                    lines.append(stripped)
                    break
                if stripped.upper() == "END":
                    break
                if not stripped and not lines:
                    continue
                lines.append(line)
        except KeyboardInterrupt:
            print("\nAborted.", file=sys.stderr)
            sys.exit(130)
        text = "\n".join(lines).strip()
        if not text:
            print("No job offering provided. Aborting.", file=sys.stderr)
            sys.exit(1)
        return text

    def get(self, url: str | None) -> str:
        if url:
            try:
                return self.from_url(url)
            except Exception as e:
                print(
                    f"Scraping failed ({e.__class__.__name__}: {e}). Falling back to manual paste.",
                    file=sys.stderr,
                )
        text = self.from_stdin()
        first_line = text.splitlines()[0].strip() if text else ""
        if first_line.startswith(("http://", "https://")) and " " not in first_line and len(text.split()) <= 2:
            print(f"Detected URL in pasted input. Crawling {first_line} ...", file=sys.stderr)
            try:
                return self.from_url(first_line)
            except Exception as e:
                print(
                    f"Crawling failed ({e.__class__.__name__}: {e}). Re-prompting for the full job text.",
                    file=sys.stderr,
                )
                return self.from_stdin()
        return text

    @staticmethod
    def _dump(path: Path, text: str) -> None:
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            print(f"Warning: could not write {path} ({e}).", file=sys.stderr)

    def _dump_all(self, result: ExtractionResult) -> None:
        if result.crawl is None:
            return
        c = result.crawl
        self._dump(CRAWL_HTML_DUMP_PATH, f"<!-- url: {c.url} -->\n{c.raw_html}")
        self._dump(CRAWL_RAW_DUMP_PATH, f"<!-- url: {c.url} -->\n{c.raw_markdown}")
        self._dump(CRAWL_FIT_DUMP_PATH, f"<!-- url: {c.url} -->\n{c.fit_markdown}")
        if c.extracted_content:
            self._dump(CRAWL_LLM_DUMP_PATH, c.extracted_content)
        if result.final_text:
            self._dump(EXTRACT_DUMP_PATH, result.final_text)

    def _print_diagnostic(self, result: ExtractionResult) -> None:
        print(f"--- extraction diagnostic (method={result.method}) ---", file=sys.stderr)
        for note in result.notes:
            print(f"  {note}", file=sys.stderr)
        print(f"--- final text preview ({len(result.final_text)} chars) ---", file=sys.stderr)
        print(result.final_text[:PREVIEW_HEAD], file=sys.stderr)
        if len(result.final_text) > PREVIEW_HEAD:
            print(f"... [{len(result.final_text) - PREVIEW_HEAD} more chars in {EXTRACT_DUMP_PATH.resolve()}]", file=sys.stderr)
        print("------------------------------------------------------", file=sys.stderr)
