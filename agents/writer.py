"""Two-phase writer: get the style right first, then enrich with ideas.

Phase 1 (draft_style)        : produce a stylistically faithful first draft.
                               No ideator input. Short, focused prompt:
                               match the example letters' voice and ground
                               every claim in the CV.

Phase 2 (incorporate_ideas)  : take the style-matched draft and weave in 1-2
                               ideas from the strategist that the candidate
                               can genuinely claim, while preserving the
                               draft's voice.

Phase 3 (revise)             : after a critic panel run, fix the bottleneck
                               specialist's complaints first while preserving
                               the rest.

Each phase has a slim prompt focused on what it must do, not on every
forbidden pattern. The critic panel is responsible for catching style drift,
AI tells, fabrication, and weak thinking - the writer doesn't need to
remember every rule, only to do its current job well."""

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from llm import chat_extra


STYLE_SYSTEM = """You write the FIRST draft of a motivation letter in the candidate's voice.

Your only job in this phase is style fidelity:
- Detect the language of the job offering and write the letter ENTIRELY in that language.
- Imitate the example motivation letters in the static context as closely as possible: tone, length, register, sentence rhythm, vocabulary. They are the ground truth for the candidate's voice.
- Ground every claim in the CV. Do not invent employers, projects, certifications, or skills.
- Pick the 1-3 most relevant CV items for THIS posting. Do not list every skill on the CV.
- Aim for 250-400 words.

A later step will incorporate strategic ideas. Do not try to fit them in here. Just produce a clean, voice-faithful, CV-grounded draft.

Output ONLY the letter body. No preamble, no commentary, no markdown fences, no signature placeholder."""


INCORPORATE_SYSTEM = """You take an existing motivation letter draft and weave in 1-2 strategic ideas while preserving the draft's voice.

You will receive:
- the draft (style-faithful, CV-grounded)
- a list of <ideas_from_strategist>

Your job:
- For each idea, decide whether the candidate's CV or the past motivation letters in the static context show enough overlap that the candidate can genuinely claim it. If yes, candidate this idea. If no, drop it.
- Pick at most 1 or 2 candidate ideas, the best fits.
- For each kept idea, weave a SHORT concrete sketch into the existing draft of how the candidate would apply this idea to THIS posting. One or two sentences max per idea.
- Preserve everything else about the draft: voice, length, structure, vocabulary. You are enriching it, not rewriting it.
- If no ideas fit, return the draft unchanged.

Output ONLY the resulting letter body."""


REVISE_SYSTEM = """You revise a motivation letter draft using a critic panel's feedback.

You will receive:
- the previous draft
- a structured critique with per-axis scores from a panel of specialist critics, plus strengths, weaknesses, suggestions
- the bottleneck axis and which specialist owns it

Your job:
- Focus FIRST on fixing the bottleneck axis. Address every weakness and suggestion from that specialist's feedback.
- Then sweep the other weaknesses, prioritizing axes that scored lowest.
- Preserve the strengths the panel called out.
- Stay in the same language as the job offering.
- Stay grounded in the CV; do not invent skills to satisfy the critic.

Output ONLY the revised letter body."""


def _format_ideas(ideas) -> str:
    if not ideas:
        return "<ideas_from_strategist>(none — no ideas were produced)</ideas_from_strategist>\n\n"
    lines = ["<ideas_from_strategist>"]
    for i, idea in enumerate(ideas, 1):
        lines.append(f"  Idea {i}:")
        lines.append(f"    problem    : {idea.problem}")
        lines.append(f"    approach   : {idea.idea}")
        lines.append(f"    why_useful : {idea.why_useful}")
    lines.append("</ideas_from_strategist>")
    return "\n".join(lines) + "\n\n"


def _format_critique(critique: dict) -> str:
    scores = critique.get("scores", {}) or {}
    by_critic = critique.get("by_critic", {}) or {}
    bottleneck_axis = critique.get("bottleneck_axis", "")
    bottleneck_critic = critique.get("bottleneck_critic", "")
    lines = [
        f"verdict          : {critique.get('verdict')}",
        f"min_score        : {critique.get('min_score')}/10",
        f"avg_score        : {critique.get('avg_score')}/10",
        f"bottleneck       : axis={bottleneck_axis}, owned by critic={bottleneck_critic} (fix this first)",
        "per-axis scores  :",
    ]
    for axis, val in scores.items():
        lines.append(f"  {axis}: {val}/10")
    if by_critic:
        lines.append("by-critic feedback:")
        for cname, cdata in by_critic.items():
            lines.append(f"  --- critic: {cname} ---")
            for w in cdata.get("weaknesses", []) or []:
                lines.append(f"    weakness   : {w}")
            for s in cdata.get("suggestions", []) or []:
                lines.append(f"    suggestion : {s}")
    return "\n".join(lines)


class WriterAgent:
    def __init__(self, client: OpenAI, model: str, max_tokens: int = 2000, temperature: float = 0.7):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call(self, system: str, static_context: str, task_block: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{static_context}\n\n---\n\n{task_block}"},
            ],
            **chat_extra(),
        )
        return (resp.choices[0].message.content or "").strip()

    # Phase 1 (single)
    def draft_style(self, static_context: str, job_offering: str, temperature: float | None = None) -> str:
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "Write the first draft now, focused only on matching the example letters' voice and grounding in the CV."
        )
        original_temp = self.temperature
        if temperature is not None:
            self.temperature = temperature
        try:
            return self._call(STYLE_SYSTEM, static_context, task)
        finally:
            self.temperature = original_temp

    # Phase 1 (beam): generate K candidate drafts in parallel at high temp.
    # Caller ranks them with deterministic metrics and picks the best.
    def draft_style_beam(
        self,
        static_context: str,
        job_offering: str,
        k: int = 5,
        temperature: float = 0.9,
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=max(k, 1)) as pool:
            results = list(
                pool.map(
                    lambda _: self.draft_style(static_context, job_offering, temperature=temperature),
                    range(k),
                )
            )
        return [r for r in results if r]

    # Phase 2
    def incorporate_ideas(
        self,
        static_context: str,
        job_offering: str,
        style_draft: str,
        ideas,
    ) -> str:
        if not ideas:
            return style_draft
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            "<existing_draft>\n"
            f"{style_draft.strip()}\n"
            "</existing_draft>\n\n"
            f"{_format_ideas(ideas)}"
            "Weave in 1 or 2 well-fitting ideas with concrete sketches, preserving the draft's voice. "
            "Output the resulting letter body."
        )
        return self._call(INCORPORATE_SYSTEM, static_context, task)

    # Phase 3
    def revise(
        self,
        static_context: str,
        job_offering: str,
        previous_draft: str,
        critique: dict,
        ideas=None,
    ) -> str:
        ideas_block = _format_ideas(ideas) if ideas else ""
        task = (
            "<job_offering>\n"
            f"{job_offering.strip()}\n"
            "</job_offering>\n\n"
            f"{ideas_block}"
            "<previous_draft>\n"
            f"{previous_draft.strip()}\n"
            "</previous_draft>\n\n"
            "<critique>\n"
            f"{_format_critique(critique)}\n"
            "</critique>\n\n"
            "Revise. Fix the bottleneck axis first. Output the revised letter body."
        )
        return self._call(REVISE_SYSTEM, static_context, task)

    # Backward-compat: a single draft() that runs phase 1 + phase 2.
    def draft(self, static_context: str, job_offering: str, ideas=None) -> str:
        style = self.draft_style(static_context, job_offering)
        return self.incorporate_ideas(static_context, job_offering, style, ideas) if ideas else style
