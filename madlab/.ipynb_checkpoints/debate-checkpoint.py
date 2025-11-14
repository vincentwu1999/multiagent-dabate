# madlab/debate.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import regex as re

from .llm import LLMClient

@dataclass
class Agent:
    name: str
    system_prompt: str
    llm: LLMClient
    reliability: float = 0.5
    id: int = field(default_factory=lambda: 0)

    def chat(self, user_prompt: str) -> str:
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}]
        return self.llm.chat(msgs).strip()

class Debate:
    def __init__(
        self,
        agents: List[Agent],
        task_type: str,
        start_prompt: str,
        debate_prompt_tmpl: str,
        rag_context: Optional[str] = None,
        rounds: int = 2,
        summarization_chars: int = 3000,
        weight_fn: Optional[Callable] = None,
        weight_ctx: Optional[Dict[str, Any]] = None,
        use_weighted: bool = True,  # <— you can disable weighted voting
    ):
        self.agents = agents
        self.task_type = task_type
        self.start_prompt = start_prompt
        self.debate_prompt_tmpl = debate_prompt_tmpl
        self.rag_context = rag_context
        self.rounds = rounds
        self.summarization_chars = summarization_chars
        self.history: List[Dict[str, Any]] = []
        self.weight_fn = weight_fn
        self.weight_ctx = weight_ctx or {}
        self.use_weighted = use_weighted

    # very light summarizer (kept here; you may replace with your own policy module)
    def _summarize(self, text: str, target_chars: int = 1500) -> str:
        if len(text) <= target_chars:
            return text
        sents = re.split(r"(?<=[\.\?\!])\s+", text)
        sents = sorted(sents, key=lambda s: -len(s))
        acc, out = 0, []
        for s in sents:
            if acc + len(s) + 1 > target_chars:
                continue
            out.append(s); acc += len(s) + 1
            if acc >= target_chars:
                break
        return " ".join(out) if out else text[:target_chars]

    def run(self) -> Dict[str, Any]:
        # Round 0: independent answers
        responses = []
        for a in self.agents:
            p = self.start_prompt
            if self.rag_context:
                p = f"Use the following evidence when useful:\n{self.rag_context}\n\n{p}"
            r = a.chat(p)
            responses.append(r)
        self.history.append({"round": 0, "responses": responses[:]})

        # Debate rounds
        for rd in range(1, self.rounds + 1):
            new_responses = []
            for i, a in enumerate(self.agents):
                others = [responses[j] for j in range(len(self.agents)) if j != i]
                other_text = "\n\n---\n".join(others)
                other_text = self._summarize(other_text, target_chars=self.summarization_chars)
                p = self.debate_prompt_tmpl.replace("{other}", other_text)
                r = a.chat(p)
                new_responses.append(r)
            responses = new_responses
            self.history.append({"round": rd, "responses": responses[:]})

        final_majority = self.aggregate_majority(responses)
        final_weighted = self.aggregate_weighted(responses, fallback=final_majority)
        return {
            "history": self.history,
            "final_majority": final_majority,
            "final_weighted": final_weighted,
            "all_final": responses,
        }

    # ------------ Aggregation ------------
    def _normalize_answer(self, text: str) -> str:
        t = text.strip()
        if self.task_type == "arithmetic":
            nums = re.findall(r"[-+]?\d+", t)
            return nums[-1] if nums else t
        if self.task_type == "gsm8k":
            m = re.search(r"\\boxed\{([^}]+)\}", t)
            if m:
                return m.group(1)
            nums = re.findall(r"[-+]?\d*\.?\d+", t)
            return nums[-1] if nums else t
        if self.task_type == "mmlu":
            m = re.search(r"\(([ABCD])\)\s*$", t.strip())
            return m.group(1) if m else t
        if self.task_type == "chess_move":
            m = re.search(r"\b14\.\s*([a-hKQRBN0O\-\=x\+#!][a-h1-8O\-x=+#]*)", t)
            return m.group(1) if m else t
        if self.task_type == "chess_valid":
            m = re.search(r"\(([a-h][1-8])\)", t, flags=re.I)
            if m: return m.group(1).lower()
            m2 = re.findall(r"\b([a-h][1-8])\b", t, flags=re.I)
            return m2[-1].lower() if m2 else t
        if self.task_type == "bio":
            lines = [ln.strip("-• ").strip() for ln in t.splitlines() if ln.strip()]
            return " | ".join(sorted(set(lines)))[:400]
        return t

    def aggregate_majority(self, responses: List[str]) -> str:
        key = [self._normalize_answer(r) for r in responses]
        counts: Dict[str, int] = {}
        for k in key:
            counts[k] = counts.get(k, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for r in responses[::-1]:
            if self._normalize_answer(r) == best:
                return r
        return responses[0]

    def aggregate_weighted(self, responses: List[str], fallback: str) -> str:
        # Optional: bypass weighted voting entirely
        if not self.use_weighted:
            return fallback

        norm = [self._normalize_answer(r) for r in responses]
        # Build weights
        if self.weight_fn is not None:
            try:
                w = self.weight_fn(self.agents, responses, self.task_type, self.weight_ctx)
            except Exception:
                w = [max(1e-3, float(getattr(a, "reliability", 0.5))) for a in self.agents]
        else:
            w = [max(1e-3, float(getattr(a, "reliability", 0.5))) for a in self.agents]

        scores: Dict[str, float] = {}
        for rr, ww in zip(norm, w):
            scores[rr] = scores.get(rr, 0.0) + float(ww)
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for r in responses[::-1]:
            if self._normalize_answer(r) == best:
                return r
        return fallback
