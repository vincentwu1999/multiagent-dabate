from dataclasses import dataclass, field
import random
from typing import List
from .llm import LLMClient

@dataclass
class Agent:
    name: str
    system_prompt: str
    llm: LLMClient
    reliability: float = 0.5
    id: int = field(default_factory=lambda: random.randint(0, 1_000_000))

    def chat(self, user_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.llm.chat(msgs).strip()

PERSONAS_BASE = [
    ("Generalist",        "You are a careful, detail-oriented analyst. Think step by step."),
    ("Historian",         "You are a historian. You check dates, sources, and historical consistency."),
    ("Engineer",          "You are a pragmatic engineer. You verify units, constraints, and edge cases."),
    ("Mathematician",     "You are a mathematician. You reason formally and verify each derivation."),
    ("Skeptical Reviewer","You are a skeptical reviewer. You aggressively fact-check and call out errors."),
]

def make_agents(llm: LLMClient, n_agents: int, persona_diversity: bool) -> List[Agent]:
    if persona_diversity:
        pool = PERSONAS_BASE[:]
        while n_agents > len(pool):
            pool.append((f"Analyst{len(pool)+1}", "You are a careful analyst."))
        roles = pool[:n_agents]
    else:
        roles = [("Analyst", "You are a careful, detail-oriented assistant. Answer concisely.")] * n_agents
    return [Agent(name=n, system_prompt=s, llm=llm) for n, s in roles]
