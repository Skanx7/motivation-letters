import json
from dataclasses import dataclass

from openai import OpenAI

from llm import chat_extra

IDEATOR_SYSTEM = """You are a senior practitioner reading a job posting. Identify the concrete TECHNICAL problems, challenges, or objectives the role implies, and propose specific TECHNICAL approaches that would help with each.

You DO NOT have access to any specific candidate's CV. Think the way a senior engineer would think about the role over coffee with a colleague.

For each idea:
- "problem" = a specific technical challenge implied by the missions or context. Not "improve fraud detection". Something like "imbalanced classes in fraud labels with concept drift over time".
- "idea" = a concrete technical approach. Name actual techniques, models, libraries, architectures. "Train an XGBoost on tabular features plus embeddings from a siamese network on claim descriptions" is good. "Apply state-of-the-art ML" or "leverage AI" is rejected.
- "why_useful" = one sentence explaining why this fits THIS role's constraints. Not a generic benefit.

Style:
- Conversational and direct, not a press release. Plain words. No marketing vocabulary like "leverage", "industrialiser", "robust", "cutting-edge", "expertise", "directement transférable", "au cœur de", "à la croisée de". No metaphors like "un pont entre".
- Technical terms are welcome and encouraged. Use real names: PyTorch, XGBoost, RAG, fine-tuning, ViT, contrastive learning, calibration, drift detection, etc.
- Be specific. "Use embeddings of constats as features for an XGBoost" beats "Use modern NLP techniques".

Aim for 3 to 5 ideas, ranked by how impactful they would be for this exact role. Skip filler ideas. If the posting is thin, fewer ideas are fine.

Return a JSON object in the language of the job posting, matching this schema and NOTHING else:
{
  "ideas": [
    {
      "problem": str,
      "idea": str,
      "why_useful": str
    }
  ]
}"""


@dataclass
class Idea:
    problem: str
    idea: str
    why_useful: str

    def to_text(self) -> str:
        return f"[{self.problem}] -> {self.idea}  // {self.why_useful}"


class IdeatorAgent:
    def __init__(self, client: OpenAI, model: str, max_tokens: int = 2000, temperature: float = 0.7):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def brainstorm(self, job_offering: str) -> list[Idea]:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": IDEATOR_SYSTEM},
                {
                    "role": "user",
                    "content": f"<job_offering>\n{job_offering.strip()}\n</job_offering>\n\nBrainstorm now.",
                },
            ],
            **chat_extra(),
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1 or e <= s:
                return []
            try:
                data = json.loads(raw[s : e + 1])
            except json.JSONDecodeError:
                return []
        items = data.get("ideas") or []
        out: list[Idea] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            problem = str(it.get("problem", "")).strip()
            idea = str(it.get("idea", "")).strip()
            why = str(it.get("why_useful", "")).strip()
            if not (problem and idea):
                continue
            out.append(Idea(problem=problem, idea=idea, why_useful=why))
        return out
