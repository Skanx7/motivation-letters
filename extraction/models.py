from dataclasses import dataclass, field


@dataclass
class JobPosting:
    title: str = ""
    company: str = ""
    location: str = ""
    contract_type: str = ""  # "Stage", "CDI", "Internship", "Full-time", ...
    duration: str = ""        # "6 mois", "12 months"
    salary: str = ""
    remote_policy: str = ""
    description: str = ""     # free-text overview / context
    missions: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    benefits: list[str] = field(default_factory=list)
    language: str = ""        # "fr", "en", "es", ...

    def has_substantive_content(self) -> bool:
        """True iff the posting has enough material for a writer to draft from."""
        if not (self.description or self.missions or self.requirements):
            return False
        return len(self.to_text()) > 200

    def to_text(self) -> str:
        """Format as a plain-text block the writer agent can ingest as `job_offering`."""
        parts: list[str] = []
        if self.title:
            parts.append(f"Titre / Title: {self.title}")
        if self.company:
            parts.append(f"Entreprise / Company: {self.company}")
        if self.location:
            parts.append(f"Lieu / Location: {self.location}")
        if self.contract_type:
            ctr = self.contract_type
            if self.duration:
                ctr = f"{ctr} ({self.duration})"
            parts.append(f"Contrat / Contract: {ctr}")
        elif self.duration:
            parts.append(f"Durée / Duration: {self.duration}")
        if self.salary:
            parts.append(f"Salaire / Salary: {self.salary}")
        if self.remote_policy:
            parts.append(f"Télétravail / Remote: {self.remote_policy}")
        if self.description:
            parts.append("\nDescription:\n" + self.description.strip())
        if self.missions:
            parts.append("\nMissions / Responsibilities:")
            parts.extend(f"- {m}" for m in self.missions)
        if self.requirements:
            parts.append("\nProfil / Requirements:")
            parts.extend(f"- {r}" for r in self.requirements)
        if self.benefits:
            parts.append("\nAvantages / Benefits:")
            parts.extend(f"- {b}" for b in self.benefits)
        return "\n".join(parts).strip()

    def to_json_dict(self) -> dict:
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "contract_type": self.contract_type,
            "duration": self.duration,
            "salary": self.salary,
            "remote_policy": self.remote_policy,
            "description": self.description,
            "missions": self.missions,
            "requirements": self.requirements,
            "benefits": self.benefits,
            "language": self.language,
        }
