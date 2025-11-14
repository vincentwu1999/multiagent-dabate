from dataclasses import dataclass
from typing import List, Dict, Optional
import os

from .config import DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS

@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS

    def __post_init__(self):
        self.use_openai = bool(os.getenv("OPENAI_API_KEY"))
        self._openai_ready = False
        if self.use_openai:
            self._init_openai()

    def _init_openai(self):
        try:
            from openai import OpenAI
            self._client = OpenAI()
            self._mode = "v1"
            self._openai_ready = True
        except Exception:
            try:
                import openai
                self._client = openai
                self._mode = "legacy"
                self._openai_ready = True
            except Exception:
                self._openai_ready = False
                self.use_openai = False

    def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        if self.use_openai and self._openai_ready:
            try:
                if getattr(self, "_mode", "v1") == "v1":
                    r = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=stop,
                    )
                    return r.choices[0].message.content
                else:
                    r = self._client.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=stop,
                    )
                    return r["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[WARN] OpenAI call failed: {e}. Falling back to mock.")
        return self._mock_chat(messages)

    # Deterministic mock so benchmarks run without keys
    def _mock_chat(self, messages: List[Dict[str, str]]) -> str:
        from .utils import int_re
        content = " ".join(m["content"].lower() for m in messages if m["role"] != "system")
        # arithmetic
        import regex as re
        m = re.search(r"what is the result of\s*([0-9+\-*\s]+)\??", content)
        if m:
            expr = m.group(1)
            try:
                val = int(eval(expr))
            except Exception:
                val = 0
            return f"My answer is {val}."
        # gsm8k
        if "your final answer should be a single numerical number" in content:
            nums = [int(x) for x in int_re.findall(content)]
            ans = sum(nums[:3]) if nums else 0
            return f"The answer is \\boxed{{{ans}}}."
        # mmlu
        import regex as re2
        if re2.search(r"\([ABCD]\)", content):
            choice = ["A","B","C","D"][abs(hash(content)) % 4]
            return f"I choose ({choice})."
        # chess move validity
        if "give one valid destination square" in content:
            return "(e4)"
        # bio
        if "bullet point biography" in content:
            return "- Person is a computer scientist.\n- Known for algorithms.\n- Received awards."
        return "Here is my updated answer. (C)"
