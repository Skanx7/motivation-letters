"""One-shot final humanizing pass.

After the critic-revise loop converges (or maxes out), the best draft may
still feel slightly mechanical even with high panel scores. This agent does
ONE pass to:
- add 1-3 natural filler words / interjections that the example letters use
  ("bref", "je pense", "j'aimerais", "du coup", "honnêtement", "en pratique")
- smooth any residual machine phrasing
- vary paragraph length so the letter reads unevenly, like real writing

It does NOT critique-loop. It is a finishing touch, not another reviewer."""

from openai import OpenAI

from llm import chat_extra

STYLIZER_SYSTEM = """You do a final stylistic pass on a motivation letter to make it sound unmistakably written by a human.

Hard rules:
- Same language as the job offering. Same length within roughly 10%. Same factual content.
- DO NOT add any new claims, skills, experiences, or technical assertions. You polish what is there; you do not invent.
- DO NOT rewrite the letter from scratch. Most sentences should remain mostly intact.

What to ADD (sparingly):
The example letters in the static context use natural fillers and interjections that real people use when writing motivation letters. Look at them. Notice phrases like "bref", "je pense", "j'aimerais", "du coup", "honnêtement", "en pratique", "concrètement", "en gros", "ce qui m'intéresse", "ce qui me parle" (or English equivalents like "honestly", "I think", "I'd like to", "basically", "in practice", "what catches my eye"). Insert 1 to 3 such markers across the WHOLE letter, only at moments where a real person writing this letter would actually use them. Do NOT sprinkle them everywhere; overdoing it is itself an AI tell. The example letters are the calibration.

What to REMOVE or REPHRASE:
- Any remaining comma-fenced apposition pattern: "X, [descriptor], Y".
- Boilerplate transitions: "De plus", "Par ailleurs", "Au-delà de", "En outre", "Furthermore", "Moreover", "Additionally". Replace with content-driven connectives or just merge sentences.
- Corporate vocabulary that does not say anything concrete: "recommandations opérationnelles", "valeur ajoutée", "stratégie data", "approche transverse", "vision business". Replace with what a person would actually say.
- Em-dashes "—" "–" or hyphens used as clause separators.
- Suspiciously uniform paragraph length. If every paragraph is the same length, vary them: a 2-line paragraph next to a 6-line one is fine.
- The "constat -> exemple -> leçon apprise" template repeated paragraph after paragraph. Break the pattern in at least one paragraph.

Output ONLY the final letter body. No preamble, no commentary, no markdown fences."""


class StylizerAgent:
    def __init__(self, client: OpenAI, model: str, max_tokens: int = 2000, temperature: float = 0.5):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def humanize(self, static_context: str, job_offering: str, draft: str) -> str:
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "<draft>\n"
            f"{draft.strip()}\n"
            "</draft>\n\n"
            "Polish the draft per the rules above. Output ONLY the final letter body."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": STYLIZER_SYSTEM},
                {"role": "user", "content": f"{static_context}\n\n---\n\n{task}"},
            ],
            **chat_extra(),
        )
        return (resp.choices[0].message.content or "").strip()
