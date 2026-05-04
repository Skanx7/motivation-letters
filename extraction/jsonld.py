"""Deterministic JobPosting extraction from schema.org JSON-LD blocks.

Most modern job boards (Welcome to the Jungle, LinkedIn, Indeed, Greenhouse,
Lever, Workable, SmartRecruiters, Workday) embed a <script type="application/ld+json">
block with @type: JobPosting. When present, it's the gold standard - structured,
unambiguous, no LLM cost, no refusal risk."""

import json
import re
from html import unescape
from typing import Iterator

from bs4 import BeautifulSoup

from .models import JobPosting


def _walk(obj) -> Iterator[dict]:
    """Yield every dict node in a nested JSON-LD payload (handles @graph, lists, nested entities)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _is_jobposting(node: dict) -> bool:
    t = node.get("@type", "")
    if isinstance(t, list):
        return any(x == "JobPosting" for x in t)
    return t == "JobPosting"


def _strip_html(text: str) -> str:
    """JobPosting.description is usually HTML. Convert to a clean plain-text block."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _flatten_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return ", ".join(_flatten_str(v) for v in val if v).strip()
    if isinstance(val, dict):
        # common shape: {"@type": "MonetaryAmount", "value": {...}}
        for key in ("name", "value", "@value", "text"):
            if key in val:
                return _flatten_str(val[key])
        return ""
    return str(val)


def _extract_location(node: dict) -> str:
    loc = node.get("jobLocation")
    if isinstance(loc, list):
        loc = loc[0] if loc else None
    if not isinstance(loc, dict):
        return ""
    addr = loc.get("address")
    if isinstance(addr, list):
        addr = addr[0] if addr else None
    if not isinstance(addr, dict):
        return _flatten_str(loc.get("name"))
    parts = [
        _flatten_str(addr.get("addressLocality")),
        _flatten_str(addr.get("addressRegion")),
        _flatten_str(addr.get("addressCountry")),
    ]
    return ", ".join(p for p in parts if p)


def _extract_salary(node: dict) -> str:
    base = node.get("baseSalary")
    if not isinstance(base, dict):
        return _flatten_str(base)
    value = base.get("value")
    if isinstance(value, dict):
        amount = _flatten_str(value.get("value")) or _flatten_str(value.get("minValue"))
        unit = _flatten_str(value.get("unitText"))
        currency = _flatten_str(base.get("currency"))
        if amount:
            return f"{amount} {currency} ({unit})".strip()
    return _flatten_str(value)


def _node_to_posting(node: dict) -> JobPosting:
    posting = JobPosting()
    posting.title = _flatten_str(node.get("title")) or _flatten_str(node.get("name"))
    org = node.get("hiringOrganization")
    if isinstance(org, dict):
        posting.company = _flatten_str(org.get("name"))
    elif isinstance(org, str):
        posting.company = org.strip()
    posting.location = _extract_location(node)
    posting.contract_type = _flatten_str(node.get("employmentType"))
    posting.salary = _extract_salary(node)
    posting.remote_policy = _flatten_str(node.get("jobLocationType")) or _flatten_str(node.get("workHours"))
    posting.description = _strip_html(_flatten_str(node.get("description")))
    posting.language = _flatten_str(node.get("inLanguage"))
    return posting


def find_jobposting(html: str) -> JobPosting | None:
    """Scan HTML for the first JSON-LD block containing a JobPosting and return it.
    Returns None if no JobPosting JSON-LD is present or all blocks fail to parse."""
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in _walk(data):
            if _is_jobposting(node):
                return _node_to_posting(node)
    return None
