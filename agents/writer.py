from openai import OpenAI

from llm import chat_extra

WRITER_SYSTEM = """You write motivation letters for the candidate whose CV appears in the static context.

Rules:
- Detect the language of the job offering and write the letter ENTIRELY in that language. Never translate, never mix languages.
- Imitate the tone, length, register, and vocabulary of the example motivation letters in the static context. They are the ground truth for the candidate's voice.
- Ground every claim in the CV. Do not invent employers, projects, certifications, or skills that are not present.
- Aim for 250-400 words. Concise, specific, no filler.
- Output ONLY the letter body. No preamble, no commentary, no markdown fences, no signature placeholders like [Your Name]."""


REVISE_INSTRUCTIONS = """Below is your previous draft and the critic's feedback. Produce a NEW version that:
- Addresses every weakness and suggestion.
- Keeps the strengths intact.
- Stays in the same language as the job offering.
- Still grounded in the CV; do not invent to satisfy the critic.

Output ONLY the revised letter body."""


class WriterAgent:
    def __init__(self, client: OpenAI, model: str, max_tokens: int = 2000, temperature: float = 0.7):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call(self, static_context: str, task_block: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": WRITER_SYSTEM},
                {"role": "user", "content": f"{static_context}\n\n---\n\n{task_block}"},
            ],
            **chat_extra(),
        )
        return (resp.choices[0].message.content or "").strip()

    def draft(self, static_context: str, job_offering: str) -> str:
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "Write the motivation letter now."
        )
        return self._call(static_context, task)

    def revise(
        self,
        static_context: str,
        job_offering: str,
        previous_draft: str,
        critique: dict,
    ) -> str:
        critique_block = (
            f"score: {critique.get('score')}/10\n"
            f"verdict: {critique.get('verdict')}\n"
            "strengths:\n  - " + "\n  - ".join(critique.get("strengths", []) or ["(none)"]) + "\n"
            "weaknesses:\n  - " + "\n  - ".join(critique.get("weaknesses", []) or ["(none)"]) + "\n"
            "suggestions:\n  - " + "\n  - ".join(critique.get("suggestions", []) or ["(none)"])
        )
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "<previous_draft>\n"
            f"{previous_draft.strip()}\n"
            "</previous_draft>\n\n"
            "<critique>\n"
            f"{critique_block}\n"
            "</critique>\n\n"
            f"{REVISE_INSTRUCTIONS}"
        )
        return self._call(static_context, task)
