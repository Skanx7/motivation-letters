from pathlib import Path

import yaml
from pypdf import PdfReader


def load_cv(path: str | Path = "cv.pdf") -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_examples(path: str | Path = "motivation_examples.yaml") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    items = raw.get("examples", [])

    # Each example is logically a (job_offering, motivation) pair. The current
    # YAML stores the two fields as separate top-level list items, so pair
    # consecutive entries; also accept the cleaner shape where a single item
    # already carries both keys.
    pairs: list[dict] = []
    pending: dict = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if "job_offering" in item and "motivation" in item:
            pairs.append({"job_offering": item["job_offering"], "motivation": item["motivation"]})
            continue
        pending.update(item)
        if "job_offering" in pending and "motivation" in pending:
            pairs.append({"job_offering": pending["job_offering"], "motivation": pending["motivation"]})
            pending = {}

    return pairs


def build_static_context(cv: str, examples: list[dict]) -> str:
    parts = ["<cv>", cv, "</cv>", ""]
    for i, ex in enumerate(examples, 1):
        parts += [
            f"<example index=\"{i}\">",
            "  <job_offering>",
            ex["job_offering"].strip(),
            "  </job_offering>",
            "  <motivation_letter>",
            ex["motivation"].strip(),
            "  </motivation_letter>",
            "</example>",
            "",
        ]
    return "\n".join(parts).strip()
