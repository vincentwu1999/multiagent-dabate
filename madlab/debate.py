# madlab/debate.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import regex as re

from .llm import LLMClient
from .prompts import PROMPTS

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
            # Find all matches and return the last one (handles confidence line after answer)
            matches = re.findall(r"\(([ABCD])\)", t.strip())
            return matches[-1] if matches else t
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

    # ------------ Full-History Debate Method ------------
    def _get_summary_agent(self) -> Agent:
        """Get or create a summary agent. Reuses first agent's LLM to minimize changes."""
        return Agent(
            name="DebateSummarizer",
            system_prompt=(
                "You summarize multi-agent debates using clustered formatting. For each round: "
                "1) Build \"Consensus Snapshot\" by grouping agents that share the same answer, reporting agent IDs, "
                "confidence min–max, and ≤150-char reasoning. 2) Record \"Divergent Opinions\" only for answers "
                "outside consensus clusters. 3) Produce a \"Change Log\" listing only agents whose answer changed or "
                "confidence shifted by ≥10% relative to the previous summary (else write None). 4) Provide "
                "\"Open Questions\" (≤150 chars) describing unresolved issues or needed follow-ups. "
                "Stay under 2500 chars, keep answers > confidence > reasoning, and preserve critical details from "
                "the previous summary while integrating new round responses."
            ),
            llm=self.agents[0].llm  # Reuse existing LLM client
        )

    def _get_full_history_prompt(self, summary: str) -> str:
        """
        Get the full-history prompt for the current task type.
        Extracts task-specific placeholders from debate_prompt_tmpl and start_prompt.
        """
        # Map task_type to full-history prompt key
        prompt_key = f"{self.task_type}_full_history"
        if prompt_key not in PROMPTS:
            # Fallback: use generic approach with summary replacement
            if "{other}" in self.debate_prompt_tmpl:
                return self.debate_prompt_tmpl.replace("{other}", f"DEBATE SUMMARY:\n{summary}\n")
            else:
                return f"DEBATE SUMMARY:\n{summary}\n\n{self.debate_prompt_tmpl}"
        
        # Get the full-history prompt template
        fh_template = PROMPTS[prompt_key]
        
        # Extract task-specific placeholders from start_prompt (required)
        # Raises ValueError if extraction fails
        template_vars = {"summary": summary}
        
        # Extract common placeholders from start_prompt (required for full-history)
        # For GSM8K: extract problem from start_prompt
        if self.task_type == "gsm8k":
            # Format: "Can you solve the following math problem?\n{problem}\nExplain..."
            # Match "problem?" (with question mark) followed by newline, then capture until "Explain"
            m = re.search(r"problem\?\s*\n(.+?)(?:\n\n|\nExplain)", self.start_prompt, re.DOTALL | re.IGNORECASE)
            if m:
                template_vars["problem"] = m.group(1).strip()
            if "problem" not in template_vars:
                raise ValueError(f"Failed to extract 'problem' from start_prompt for GSM8K task. start_prompt: {self.start_prompt[:200]}...")
        
        # Extract from start_prompt for arithmetic
        if self.task_type == "arithmetic":
            # Extract numbers from start_prompt (format: {}+{}*{}+{}-{}*{})
            nums = re.findall(r"\d+", self.start_prompt)
            if len(nums) >= 6:
                # Arithmetic prompt uses positional {} placeholders - format with summary first, then numbers
                try:
                    # Replace {summary} first, then format with positional args for numbers
                    temp_template = fh_template.replace("{summary}", summary)
                    return temp_template.format(nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])
                except Exception as e:
                    raise ValueError(f"Failed to format arithmetic prompt: {e}")
            else:
                raise ValueError(f"Failed to extract 6 numbers from start_prompt for arithmetic task. Found {len(nums)} numbers. start_prompt: {self.start_prompt[:200]}...")
        
        # Extract MMLU placeholders
        if self.task_type == "mmlu":
            # Format: "Can you answer the following question as accurately as possible?\n\n{question}\nA) {A}\n..."
            # Match after "possible?" followed by newlines, then capture until "A)"
            m = re.search(r"possible\?\s*\n\n(.+?)\nA\)", self.start_prompt, re.DOTALL | re.IGNORECASE)
            if m:
                template_vars["question"] = m.group(1).strip()
            else:
                raise ValueError(f"Failed to extract 'question' from start_prompt for MMLU task. start_prompt: {self.start_prompt[:200]}...")
            for opt in ["A", "B", "C", "D"]:
                m = re.search(rf"{opt}\)\s*([^\n]+)", self.start_prompt)
                if m:
                    template_vars[opt] = m.group(1).strip()
                else:
                    raise ValueError(f"Failed to extract option '{opt}' from start_prompt for MMLU task. start_prompt: {self.start_prompt[:200]}...")
        
        # Extract bio person
        if self.task_type == "bio":
            # Format: "Give a bullet point biography of {person} highlighting..."
            # Capture person name until first space
            m = re.search(r"biography of ([^\s]+)", self.start_prompt, re.IGNORECASE)
            if m:
                template_vars["person"] = m.group(1).strip()
            else:
                raise ValueError(f"Failed to extract 'person' from start_prompt for bio task. start_prompt: {self.start_prompt[:200]}...")
        
        # Extract chess placeholders
        if self.task_type in ["chess_valid", "chess_move"]:
            if self.task_type == "chess_valid":
                # Format: "Given the chess game\n{moves}\n give one valid..."
                # Match after "game" followed by newline, then capture until " give" or newline before "give"
                m = re.search(r"game\s*\n(.+?)(?:\n\n|\ngive)", self.start_prompt, re.DOTALL | re.IGNORECASE)
            else:  # chess_move
                # Format: "Here is the current sequence of moves in a chess game:\n{moves}\n What is..."
                # Match after "game:" followed by newline, then capture until " What" or newline before "What"
                m = re.search(r"game:\s*\n(.+?)(?:\n\n|\nWhat)", self.start_prompt, re.DOTALL | re.IGNORECASE)
            if m:
                template_vars["moves"] = m.group(1).strip()
            else:
                raise ValueError(f"Failed to extract 'moves' from start_prompt for chess task. start_prompt: {self.start_prompt[:200]}...")
            if self.task_type == "chess_valid":
                m = re.search(r"at ([a-h][1-8])", self.start_prompt, re.IGNORECASE)
                if m:
                    template_vars["origin"] = m.group(1).strip()
                else:
                    raise ValueError(f"Failed to extract 'origin' from start_prompt for chess_valid task. start_prompt: {self.start_prompt[:200]}...")
        
        # Extract bioasq placeholders
        if self.task_type == "bioasq":
            # Format: "Question: {question}\nType: {qtype}..."
            m = re.search(r"Question:\s*([^\n]+)", self.start_prompt, re.IGNORECASE)
            if m:
                template_vars["question"] = m.group(1).strip()
            else:
                raise ValueError(f"Failed to extract 'question' from start_prompt for bioasq task. start_prompt: {self.start_prompt[:200]}...")
            m = re.search(r"Type:\s*([^\n]+)", self.start_prompt, re.IGNORECASE)
            if m:
                template_vars["qtype"] = m.group(1).strip()
            else:
                raise ValueError(f"Failed to extract 'qtype' from start_prompt for bioasq task. start_prompt: {self.start_prompt[:200]}...")
        
        # Format the template with extracted variables
        try:
            return fh_template.format(**template_vars)
        except KeyError as e:
            # If missing placeholders, raise an error with details
            missing = [key for key in ["problem", "question", "person", "moves", "origin", "qtype", "A", "B", "C", "D"] 
                      if f"{{{key}}}" in fh_template and key not in template_vars]
            raise ValueError(
                f"Failed to extract required placeholders for {self.task_type} task. "
                f"Missing: {missing}. "
                f"Extracted: {list(template_vars.keys())}. "
                f"start_prompt preview: {self.start_prompt[:200]}..."
            ) from e
    
    def _format_responses_for_summary(self, responses: List[str], round_num: int) -> str:
        """Format agent responses for summary prompt. Smart truncation to preserve key info."""
        formatted = []
        # For fair comparison with regular method (which uses 3000 chars for other agents' responses):
        # Allocate ~800-1000 chars per agent response (3 agents = 2400-3000 chars total)
        # This matches the regular method's summarization_chars=3000 budget
        max_per_response = 900
        for i, resp in enumerate(responses):
            if len(resp) <= max_per_response:
                truncated = resp
            else:
                # Smart truncation: take beginning (reasoning) and end (answer)
                # Take first 60% and last 40% of allocated space
                first_part = max_per_response * 6 // 10  # ~540 chars for reasoning
                last_part = max_per_response - first_part - 10  # ~350 chars for answer/conclusion
                truncated = resp[:first_part] + "\n[...]\n" + resp[-last_part:]
            formatted.append(f"Agent {i+1}:\n{truncated}")
        result = "\n\n".join(formatted)
        # Ensure total is under 3000 chars (matching regular method's summarization_chars budget)
        # Summary prompt template is ~200 chars, so we have ~2800 chars for responses
        max_total = 2800
        if len(result) > max_total:
            # If still too long, proportionally reduce each agent's allocation
            scale = max_total / len(result)
            formatted = []
            for i, resp in enumerate(responses):
                new_max = int(max_per_response * scale)
                if len(resp) <= new_max:
                    truncated = resp
                else:
                    first_part = new_max * 6 // 10
                    last_part = new_max - first_part - 10
                    truncated = resp[:first_part] + "\n[...]\n" + resp[-last_part:]
                formatted.append(f"Agent {i+1}:\n{truncated}")
            result = "\n\n".join(formatted)
            if len(result) > max_total:
                result = result[:max_total] + "..."
        return result

    def run_full_history(self, summary_agent: Optional[Agent] = None) -> Dict[str, Any]:
        """
        Run debate with full-history summarization.
        Flow:
        - Round 0: cumulative_summary_0 = None, generate responses_0 independently
        - Round 1: summarize cumulative_summary_0 (None) + responses_0 → cumulative_summary_1,
                   then agents use cumulative_summary_1 → responses_1
        - Round 2: summarize cumulative_summary_1 + responses_1 → cumulative_summary_2,
                   then agents use cumulative_summary_2 → responses_2
        Reuses aggregate_majority() for final voting.
        Supports any number of rounds (self.rounds).
        Character limits match regular method for fair comparison:
        - Summary prompts: up to 3500 chars (matches regular method's ~3200 char prompt budget)
        - Cumulative summaries: up to 2500 chars (matches regular method's agent response length)
        - Agent prompts: up to 3500 chars (matches regular method's prompt budget)
        """
        summary_agent = summary_agent or self._get_summary_agent()
        round_responses = []  # Track rounds with summaries
        cumulative_summary = None  # Track cumulative summary (starts as None for round 0)
        
        # Round 0: cumulative_summary_0 = None, generate responses_0 independently
        responses = []
        for a in self.agents:
            p = self.start_prompt
            if self.rag_context:
                p = f"Use the following evidence when useful:\n{self.rag_context}\n\n{p}"
            r = a.chat(p)
            responses.append(r)
        
        # Store round 0 with cumulative_summary_0 = None
        round_responses.append({"round": 0, "responses": responses[:], "debate_summary": None})
        
        # Rounds 1 to N
        for rd in range(1, self.rounds + 1):
            # Step 1: Summarize previous cumulative_summary + previous round's responses → new cumulative_summary
            # This summary will be used by agents in the current round
            prev_responses = responses  # Responses from previous round
            summary_input = self._format_responses_for_summary(prev_responses, round_num=rd-1)
            
            # For fair comparison: allow previous summary similar to regular method's budget
            # Previous summary can be up to 1500 chars (half of the 3000 char budget, leaving room for new responses)
            prev_summary_truncated = cumulative_summary[:1500] if cumulative_summary and len(cumulative_summary) > 1500 else (cumulative_summary or "")
            previous_summary_text = f"\n\nPREVIOUS SUMMARY:\n{prev_summary_truncated}\n" if prev_summary_truncated else ""
            
            summary_prompt = PROMPTS["debate_full_history_summary"].format(
                round_num=rd-1,
                responses=summary_input,
                previous_summary_section=previous_summary_text
            )
            # Allow up to 3500 chars for summary prompt (matching regular method's prompt budget)
            if len(summary_prompt) > 3500:
                summary_prompt = summary_prompt[:3500]
            
            cumulative_summary = summary_agent.chat(summary_prompt)
            # Allow richer summaries: 2500 chars (similar to regular method's agent response length)
            if len(cumulative_summary) > 2500:
                cumulative_summary = cumulative_summary[:2500]
            
            # Step 2: Agents use cumulative_summary to generate new responses
            new_responses = []
            for agent in self.agents:
                # For fair comparison: allow cumulative summary similar to regular method's 3000 chars
                # Regular method: debate template (~300) + summarized other agents (3000) = ~3300 chars
                # Full-history: debate template (~300) + cumulative summary = ~3300 chars
                # So allow up to 3000 chars for cumulative summary
                summary_for_prompt = cumulative_summary[:3000] if cumulative_summary else ""
                
                # Create prompt: use full-history specific prompt template
                prompt_base = self._get_full_history_prompt(summary_for_prompt)
                
                # Allow up to 3500 chars total (matching regular method's prompt budget)
                if len(prompt_base) > 3500:
                    prompt_base = prompt_base[:3500]
                
                new_response = agent.chat(prompt_base)
                new_responses.append(new_response)
            
            responses = new_responses
            
            # Store round with the cumulative_summary that was used to generate these responses
            # For the last round, we still store the summary (even though it won't be used)
            round_responses.append({
                "round": rd,
                "responses": responses[:],
                "debate_summary": cumulative_summary
            })
        
        # Final: Use existing aggregate_majority() method (reuse!)
        final_majority = self.aggregate_majority(responses)
        
        return {
            "round_responses": round_responses,
            "final_majority": final_majority,
            "all_final": responses,
        }