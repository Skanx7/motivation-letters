import json
from dataclasses import dataclass, field

from openai import OpenAI

from llm import chat_extra

CRITIC_SYSTEM = """You are a harsh-but-fair critic of motivation letters. You evaluate a draft against the candidate's CV (in the static context), the example letter(s) that define the candidate's voice, and the target job offering.

Score (1-10) on the weighted average of:
  - job-relevance: does it speak to the job's actual requirements?
  - authenticity to CV: every claim must be traceable to the CV. Fabrication = automatic score <= 4.
  - style fit: tone, register, length, and vocabulary match the example letter(s).
  - language correctness: written in the job offering's language, idiomatic, no awkward phrasing.

Verdict rules:
  - "ship" only if score >= 8 AND no fabricated claims.
  - "revise" otherwise.

Write all string fields (strengths, weaknesses, suggestions) in the same language as the job offering, so the writer can act on them in-context. Be specific - cite phrases from the draft when possible.

Return your critique as a JSON object matching this schema and NOTHING else:
{
  "score": int 1-10,
  "verdict": "ship" | "revise",
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "suggestions": [string, ...]
}"""


@dataclass
class Critique:
    score: int
    verdict: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
        }


def _repair_truncated_json(text: str) -> str:
    """Close any unterminated string and unbalanced brackets so a truncated
    LLM JSON response (max_tokens hit mid-output) can still be parsed."""
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
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Last resort: repair an unterminated string / unbalanced brackets (truncation case).
    candidate = text[start:] if start != -1 else text
    try:
        return json.loads(_repair_truncated_json(candidate))
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse critic JSON: {e}\n--- raw output ---\n{text[:500]}") from e


class CriticAgent:
    def __init__(self, client: OpenAI, model: str, max_tokens: int = 4000, temperature: float = 0.2):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def critique(self, static_context: str, job_offering: str, draft: str) -> Critique:
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "<draft_to_critique>\n"
            f"{draft.strip()}\n"
            "</draft_to_critique>\n\n"
            "Critique the draft now. Respond with ONLY the JSON object."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"{static_context}\n\n---\n\n{task}"},
            ],
            **chat_extra(),
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _extract_json(raw)

        return Critique(
            score=int(data["score"]),
            verdict=str(data["verdict"]),
            strengths=list(data.get("strengths", [])),
            weaknesses=list(data.get("weaknesses", [])),
            suggestions=list(data.get("suggestions", [])),
        )
