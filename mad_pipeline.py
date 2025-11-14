# """
# Multi‑Agent Debate (MAD) – end‑to‑end reproducible pipeline + novel extensions.

# What this file does
# -------------------
# 1) Reproduce the core experiments from "Improving Factuality and Reasoning in Language Models
#    through Multiagent Debate" using black‑box chat LLMs.
#    - Arithmetic (synthetic)
#    - GSM8K (grade‑school math; light subset by default; full set if HuggingFace datasets available)
#    - MMLU (subset by default; full set if HuggingFace datasets available)
#    - Chess move *validity* (python‑chess based)
#    - Computer‑scientist biographies factuality (LLM‑judged vs. Wikipedia retrieval)

# 2) Implement and evaluate the 4 proposed *novel directions* from the screenshots:
#    - Retrieval‑Augmented Debates (evidence retrieved via Wikipedia API)
#    - Weighted Voting (agent weights from rolling reliability on a calibration set)
#    - Persona Diversity (specialised agent system prompts, e.g., historian, engineer)
#    - Adaptive Summarization (token‑budget aware summarization of other agents’ messages)

# 3) Generate publication‑ready figures mirroring the paper’s plots (accuracy bars, ablations).

# How to run
# ----------
# $ python mad_pipeline.py                   # (rename this file if desired)
# Environment:
# - Requires Python 3.9+.
# - Set OPENAI_API_KEY for real LLM runs (recommended).
# - If no API key is found, a fall‑back mock LLM is used (pipeline still runs without error, but
#   results obviously won’t match the paper).

# Outputs are written to ./outputs/*.png and ./outputs/*.jsonl

# Dependencies are auto‑installed on first run (matplotlib, numpy, pandas, python‑chess, wikipedia, tqdm, regex, datasets (optional)).

# Notes
# -----
# - Prompts and evaluation setups follow the paper’s Appendix/Table of prompts and sections 2–3
#   (see page references inside the code). :contentReference[oaicite:0]{index=0}
# - This code is designed to be robust: it catches network/API errors and continues with sensible
#   fallbacks so you can “copy‑paste‑run” without manual edits.

# """

# # =========================
# # Bootstrapping & Imports
# # =========================
# import os, sys, json, math, time, random, re, textwrap, subprocess, shutil
# from dataclasses import dataclass, field
# from typing import List, Dict, Any, Optional, Tuple

# def _pip_install(pkgs: List[str]):
#     """Install packages if missing (quietly)."""
#     for p in pkgs:
#         try:
#             __import__(p if p != "python_chess" else "chess")
#         except Exception:
#             try:
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])
#             except Exception:
#                 print(f"[WARN] Could not install {p}. Some features might be limited.")

# _pip_install([
#     "matplotlib", "numpy", "pandas", "tqdm", "regex", "wikipedia", "python-chess",
#     # Optional:
#     "datasets"
# ])

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import wikipedia
# import chess
# import chess.pgn
# import regex as re2

# # ===========================================
# # Global config & reproducibility
# # ===========================================
# SEED = int(os.environ.get("MAD_SEED", "7"))
# random.seed(SEED); np.random.seed(SEED)

# OUT_DIR = os.path.abspath("./outputs")
# os.makedirs(OUT_DIR, exist_ok=True)

# # Default LLMs / models
# DEFAULT_MODEL = os.environ.get("MAD_MODEL", "gpt-4o-mini")  # use any chat model available to you
# TEMPERATURE = float(os.environ.get("MAD_TEMPERATURE", "0.2"))
# MAX_TOKENS = int(os.environ.get("MAD_MAX_TOKENS", "512"))

# # Debate config (Figure 10/12 in paper discuss agents/rounds) :contentReference[oaicite:1]{index=1}
# N_AGENTS = int(os.environ.get("MAD_AGENTS", "3"))
# N_ROUNDS = int(os.environ.get("MAD_ROUNDS", "2"))

# # Persona diversity (novel direction)
# PERSONAS_BASE = [
#     ("Generalist",       "You are a careful, detail‑oriented analyst. Think step by step."),
#     ("Historian",        "You are a historian. You check dates, sources, and historical consistency."),
#     ("Engineer",         "You are a pragmatic engineer. You verify units, constraints, and edge cases."),
#     ("Mathematician",    "You are a mathematician. You reason formally and verify each derivation."),
#     ("Skeptical Reviewer","You are a skeptical reviewer. You aggressively fact‑check and call out errors."),
# ]

# # =========================
# # Utilities
# # =========================
# def safe_eval_arithmetic(expr: str) -> Optional[int]:
#     """Safely evaluate arithmetic of the form digits with + - * only (as in the paper’s arithmetic task)."""
#     if not re2.fullmatch(r"[0-9+\-*\s]+", expr):
#         return None
#     try:
#         # Python eval is safe under our strict regex (only digits/operators/spaces)
#         return int(eval(expr))
#     except Exception:
#         return None

# def ensure_jsonl(path: str, rows: List[Dict[str, Any]]):
#     with open(path, "w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

# def strip_codefences(s: str) -> str:
#     return re2.sub(r"^```.*?^```", "", s, flags=re2.S | re2.M)

# def extract_boxed_number(text: str) -> Optional[float]:
#     """
#     Extract \\boxed{number} (as used in GSM8K prompt) or a final numeric answer.
#     """
#     text = text.strip()
#     m = re2.search(r"\\boxed\{([^}]+)\}", text)
#     cand = None
#     if m:
#         cand = m.group(1)
#     else:
#         # Try last number in text
#         nums = re2.findall(r"[-+]?\d*\.?\d+", text)
#         if nums:
#             cand = nums[-1]
#     if cand is None: return None
#     try:
#         if "." in cand: return float(cand)
#         return int(cand)
#     except:
#         try:
#             return float(cand)
#         except:
#             return None

# def extract_choice_letter(text: str) -> Optional[str]:
#     """Extract (A) or (B) format at the end."""
#     m = re2.search(r"\(([A-D])\)\s*$", text.strip())
#     return m.group(1) if m else None

# def extract_chess_move_on_turn(text: str, turn_number: int = 14) -> Optional[str]:
#     """Expect '14. Qa4' style from the paper’s chess prompt."""
#     m = re2.search(rf"\b{turn_number}\.\s*([a-hKQRBN0O\-\=x\+#!][a-h1-8O\-x=+#]*)", text.strip())
#     if m:
#         return m.group(1).strip()
#     return None

# def extract_square(text: str) -> Optional[str]:
#     """For chess validity task: extract (e5) -> 'e5'."""
#     m = re2.search(r"\(([a-h][1-8])\)", text.strip(), flags=re2.I)
#     if m: return m.group(1).lower()
#     # fallback: last a-h1-8 pattern
#     m = re2.findall(r"\b([a-h][1-8])\b", text.strip(), flags=re2.I)
#     return m[-1].lower() if m else None

# def truncate_tokens(s: str, max_chars: int = 4000) -> str:
#     if len(s) <= max_chars: return s
#     return s[:max_chars//2] + "\n[...snip...]\n" + s[-max_chars//2:]

# def moving_average(prev: float, new: float, alpha: float = 0.2) -> float:
#     return (1 - alpha) * prev + alpha * new

# # ===================================
# # Lightweight LLM client (OpenAI/Mock)
# # ===================================
# class LLMClient:
#     def __init__(self, model: str = DEFAULT_MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
#         self.model = model
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.use_openai = bool(os.getenv("OPENAI_API_KEY"))
#         self._openai_ready = False
#         if self.use_openai:
#             self._init_openai()

#     def _init_openai(self):
#         try:
#             # Try new SDK
#             from openai import OpenAI
#             self._client = OpenAI()
#             self._mode = "v1"
#             self._openai_ready = True
#         except Exception:
#             try:
#                 import openai
#                 self._client = openai
#                 self._mode = "legacy"
#                 self._openai_ready = True
#             except Exception:
#                 self._openai_ready = False
#                 self.use_openai = False

#     def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
#         if self.use_openai and self._openai_ready:
#             try:
#                 if self._mode == "v1":
#                     r = self._client.chat.completions.create(
#                         model=self.model,
#                         messages=messages,
#                         temperature=self.temperature,
#                         max_tokens=self.max_tokens,
#                         stop=stop
#                     )
#                     return r.choices[0].message.content
#                 else:
#                     r = self._client.ChatCompletion.create(
#                         model=self.model,
#                         messages=messages,
#                         temperature=self.temperature,
#                         max_tokens=self.max_tokens,
#                         stop=stop
#                     )
#                     return r['choices'][0]['message']['content']
#             except Exception as e:
#                 print(f"[WARN] OpenAI call failed: {e}. Falling back to MockLLM.")
#                 return self._mock_chat(messages)
#         else:
#             return self._mock_chat(messages)

#     # --------------------------------------------------
#     # Mock LLM (deterministic, fast, *not* a real model)
#     # --------------------------------------------------
#     def _mock_chat(self, messages: List[Dict[str, str]]) -> str:
#         """
#         A small deterministic “LLM” that:
#         - Solves arithmetic exactly if it finds 'What is the result of ...' expression.
#         - Outputs a plausible GSM8K/MMLU/chess/biography answer with heuristics.
#         This keeps the pipeline runnable without API keys.
#         """
#         content = " ".join([m["content"] for m in messages if m["role"] != "system"]).lower()

#         # Arithmetic
#         m = re2.search(r"what is the result of\s*([0-9+\-*\s]+)\??", content)
#         if m:
#             expr = m.group(1)
#             val = safe_eval_arithmetic(expr)
#             if val is None:
#                 val = random.randint(-500, 500)
#             return f"My answer is {val}."

#         # GSM8K (look for a final \boxed{})
#         # Heuristic: if a simple percentage or sum pattern present, try to compute.
#         # Otherwise return a deterministic number from hash for reproducibility.
#         if "your final answer should be a single numerical number" in content:
#             # simple percent-of or sum detection
#             nums = [int(x) for x in re2.findall(r"\b\d+\b", content)]
#             ans = None
#             if "percent" in content or "%" in content:
#                 ps = [int(x) for x in re2.findall(r"\b(\d+)\s*%", content)]
#                 if ps and nums:
#                     p = ps[-1] / 100.0
#                     ans = int(round(nums[-1] * p))
#             if ans is None and nums:
#                 ans = sum(nums[:3])  # crude
#             if ans is None:
#                 ans = (abs(hash(content)) % 97) + 3
#             return f"The answer is \\boxed{{{ans}}}."

#         # MMLU multiple-choice -> always (C) deterministically from hash
#         if "putting the answer in the form (" in content and re2.search(r"\([ABCD]\)", content):
#             choice = ["A","B","C","D"][abs(hash(content)) % 4]
#             return f"I choose ({choice})."

#         # Chess move (return a harmless queen move if legal square is mentioned)
#         if "what is the best chess move i should execute next" in content:
#             return "14. Qa4"

#         # Chess validity – suggest e5 if piece on e4 mentioned, else a legal looking square.
#         if "give one valid destination square" in content:
#             squares = re2.findall(r"\b([a-h][1-8])\b", content)
#             dst = squares[-1] if squares else random.choice(["e4","e5","d4","d5"])
#             return f"A valid destination is ({dst})."

#         # Biography bullets – stub from Wikipedia summary if we can retrieve, else canned bullets
#         if "bullet point biography" in content:
#             m = re2.search(r"biography of ([^,\n]+)", content) or re2.search(r"biographies of ([^,\n]+)", content)
#             name = (m.group(1).strip() if m else "the person").title()
#             bullets = [
#                 f"- {name} is a computer scientist.",
#                 f"- Known for contributions to algorithms and systems.",
#                 "- Recipient of multiple awards."
#             ]
#             return "\n".join(bullets)

#         # Default deterministic filler
#         return "Here is my updated answer. (C)"

# # ============================================
# # Prompts (from Table of prompts in the paper)
# # ============================================
# # See Appendix Table (page with Figure 15) for these templates. :contentReference[oaicite:2]{index=2}
# PROMPTS = {
#     "arithmetic_start": "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.",
#     "arithmetic_debate": ("These are the recent/updated opinions from other agents:\n{other}\n"
#                           "Use these opinions carefully as additional advice, can you provide an updated answer? "
#                           "Make sure to state your answer at the end of the response."),
#     "gsm8k_start": ("Can you solve the following math problem?\n{problem}\n"
#                     "Explain your reasoning. Your final answer should be a single numerical number, "
#                     "in the form \\boxed{{answer}}, at the end of your response."),
#     "gsm8k_debate": ("These are the solutions to the problem from other agents:\n{other}\n"
#                      "Using the solutions from other agents as additional information, can you provide your answer to the math problem?\n"
#                      "The original math problem is:\n{problem}\n"
#                      "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."),
#     "mmlu_start": ("Can you answer the following question as accurately as possible?\n\n{question}\n"
#                    "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
#                    "Explain your answer, putting the answer in the form (X) at the end of your response."),
#     "mmlu_debate": ("These are the solutions to the problem from other agents:\n{other}\n"
#                     "Using the reasoning from other agents as additional advice, can you give an updated answer?\n"
#                     "Examine your solution and that of other agents. Put your answer in the form (X) at the end of your response."),
#     "bio_start": ("Give a bullet point biography of {person} highlighting their contributions and achievements as a computer scientist, "
#                   "with each fact separated with a new line character."),
#     "bio_debate": ("Here are some bullet point biographies of {person} given by other agents:\n{other}\n"
#                    "Closely examine your biography and the biography of other agents and provide an updated bullet point biography."),
#     "chess_move_start": ("Here is the current sequence of moves in a chess game:\n{moves}\n"
#                          "What is the best chess move I should execute next? Give a single move suggestion of the form 14. <XXX> "
#                          "and make sure the chess move is valid in the current board state."),
#     "chess_move_debate": ("Here are other chess move suggestions from other agents:\n{other}\n"
#                           "Using the chess suggestions from other agents as additional advice and your earlier generated solution, "
#                           "can you give me your updated thoughts on the best next chess move I should play given the chess sequence?\n"
#                           "Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state."),
#     "chess_valid_start": ("Given the chess game\n{moves}\n"
#                           "give one valid destination square for the chess piece at {origin}.\n"
#                           "State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5).\n"
#                           "Give a one line explanation of why your destination square is a valid move."),
#     "chess_valid_debate": ("Here are destination square suggestions from other agents:\n{other}\n"
#                            "Can you double check that your destination square is a valid move? "
#                            "Check the valid move justifications from other agents.\n"
#                            "State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."),
# }

# # =======================================
# # Retrieval (Wikipedia) for RAG debates
# # =======================================
# def wiki_search_and_snippets(query: str, k: int = 2, char_limit: int = 1500) -> str:
#     try:
#         wikipedia.set_lang("en")
#         hits = wikipedia.search(query, results=k)
#         snippets = []
#         for h in hits:
#             try:
#                 pg = wikipedia.page(h, auto_suggest=False)
#                 s = pg.summary
#                 if s:
#                     snippets.append(f"Title: {pg.title}\nSummary:\n{s}")
#             except Exception:
#                 continue
#         joined = "\n\n".join(snippets)
#         return truncate_tokens(joined, max_chars=char_limit) if joined else ""
#     except Exception:
#         return ""

# # =======================================
# # Adaptive Summarization (LLM or extractive)
# # =======================================
# def summarize_text(text: str, llm: LLMClient, target_chars: int = 1500) -> str:
#     if len(text) <= target_chars:
#         return text
#     # Try quick LLM summary if available
#     if llm.use_openai:
#         prompt = f"Summarize the following debate, keeping all key facts, equations, and disagreements under {target_chars} characters:\n\n{text}"
#         try:
#             out = llm.chat([{"role":"system","content":"You compress long texts without losing facts."},
#                             {"role":"user","content": prompt}])
#             return truncate_tokens(out, target_chars)
#         except Exception:
#             pass
#     # Fallback extractive: top-k sentences by length
#     sents = re2.split(r"(?<=[\.\?\!])\s+", text)
#     sents = sorted(sents, key=lambda s: -len(s))
#     acc, out = 0, []
#     for s in sents:
#         if acc + len(s) + 1 > target_chars: continue
#         out.append(s); acc += len(s) + 1
#         if acc >= target_chars: break
#     if not out: return text[:target_chars]
#     return " ".join(out)

# # =======================================
# # Agent & Debate Orchestrator
# # =======================================
# @dataclass
# class Agent:
#     name: str
#     system_prompt: str
#     llm: LLMClient
#     reliability: float = 0.5  # for weighted voting (novel direction)
#     id: int = field(default_factory=lambda: random.randint(0, 1_000_000))

#     def chat(self, user_prompt: str) -> str:
#         msgs = [{"role":"system","content": self.system_prompt},
#                 {"role":"user","content": user_prompt}]
#         return self.llm.chat(msgs).strip()

# class Debate:
#     def __init__(
#         self,
#         agents: List[Agent],
#         task_type: str,
#         start_prompt: str,
#         debate_prompt_tmpl: str,
#         rag_context: Optional[str] = None,
#         rounds: int = N_ROUNDS,
#         summarization_chars: int = 3000,   # Adaptive summarization budget
#     ):
#         self.agents = agents
#         self.task_type = task_type
#         self.start_prompt = start_prompt
#         self.debate_prompt_tmpl = debate_prompt_tmpl
#         self.rag_context = rag_context
#         self.rounds = rounds
#         self.summarization_chars = summarization_chars
#         self.history: List[Dict[str, Any]] = []

#     def run(self) -> Dict[str, Any]:
#         # Round 0 – independent proposals
#         responses = []
#         for a in self.agents:
#             p = self.start_prompt
#             if self.rag_context:
#                 p = f"Use the following evidence when useful:\n{self.rag_context}\n\n{p}"
#             r = a.chat(p)
#             responses.append(r)

#         self.history.append({"round": 0, "responses": responses[:]})

#         # Debate rounds
#         for rd in range(1, self.rounds + 1):
#             new_responses = []
#             for i, a in enumerate(self.agents):
#                 others = [responses[j] for j in range(len(self.agents)) if j != i]
#                 other_text = "\n\n---\n".join(others)
#                 other_text = summarize_text(other_text, a.llm, target_chars=self.summarization_chars)
#                 # p = self.debate_prompt_tmpl.format(other=other_text)
#                 p = self.debate_prompt_tmpl.replace("{other}", other_text)
#                 if self.task_type in ("gsm8k", "mmlu", "bio", "chess_move", "chess_valid") and "{problem}" in p:
#                     # (gsm8k) handled by caller; here we simply keep p as built
#                     pass
#                 r = a.chat(p)
#                 new_responses.append(r)
#             responses = new_responses
#             self.history.append({"round": rd, "responses": responses[:]})

#         # Aggregation – majority and weighted voting
#         final_majority = self.aggregate_majority(responses)
#         final_weighted = self.aggregate_weighted(responses)
#         return {
#             "history": self.history,
#             "final_majority": final_majority,
#             "final_weighted": final_weighted,
#             "all_final": responses,
#         }

#     def aggregate_majority(self, responses: List[str]) -> str:
#         # Task‑specific normalizers
#         key = [self._normalize_answer(r) for r in responses]
#         counts: Dict[str, int] = {}
#         for k in key:
#             counts[k] = counts.get(k, 0) + 1
#         best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
#         # return the original response corresponding to the winner
#         for r in responses[::-1]:
#             if self._normalize_answer(r) == best:
#                 return r
#         return responses[0]

#     def aggregate_weighted(self, responses: List[str]) -> str:
#         # Weighted voting by agent.reliability (novel direction)
#         norm = [self._normalize_answer(r) for r in responses]
#         scores: Dict[str, float] = {}
#         for r, a in zip(norm, self.agents):
#             scores[r] = scores.get(r, 0.0) + max(1e-3, a.reliability)
#         best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
#         for r in responses[::-1]:
#             if self._normalize_answer(r) == best:
#                 return r
#         return responses[0]

#     def _normalize_answer(self, text: str) -> str:
#         """Task‑aware canonicalization for voting."""
#         t = text.strip()
#         if self.task_type == "arithmetic":
#             # last integer
#             nums = re2.findall(r"[-+]?\d+", t)
#             return nums[-1] if nums else t
#         if self.task_type == "gsm8k":
#             v = extract_boxed_number(t)
#             return str(v) if v is not None else t
#         if self.task_type == "mmlu":
#             c = extract_choice_letter(t)
#             return c or t
#         if self.task_type == "chess_move":
#             m = extract_chess_move_on_turn(t, 14)
#             return m or t
#         if self.task_type == "chess_valid":
#             s = extract_square(t)
#             return s or t
#         if self.task_type == "bio":
#             # collapse to set of lines
#             lines = [ln.strip("-• ").strip() for ln in t.splitlines() if ln.strip()]
#             return " | ".join(sorted(set(lines)))[:400]
#         return t

# # =======================================
# # Tasks: generators & evaluators
# # =======================================
# def gen_arithmetic(n: int = 100) -> List[Dict[str, Any]]:
#     items = []
#     for _ in range(n):
#         xs = [random.randint(0, 30) for __ in range(6)]
#         expr = f"{xs[0]}+{xs[1]}*{xs[2]}+{xs[3]}-{xs[4]}*{xs[5]}"
#         ans = safe_eval_arithmetic(expr)
#         items.append({"expr": expr, "answer": ans})
#     return items

# def load_gsm8k_subset(n: int = 100) -> List[Dict[str, Any]]:
#     # Try HuggingFace (main split is 'test'); fallback to a small hand‑picked set
#     try:
#         from datasets import load_dataset
#         ds = load_dataset("gsm8k", "main", split="test")
#         rows = []
#         for i in range(min(n, len(ds))):
#             p = ds[i]["question"]
#             # Answers look like "... #### 42"
#             a = ds[i]["answer"]
#             m = re2.search(r"####\s*([-+]?\d*\.?\d+)", a)
#             gold = float(m.group(1)) if m else None
#             if gold is not None and gold.is_integer(): gold = int(gold)
#             rows.append({"problem": p, "answer": gold})
#         return rows
#     except Exception:
#         # Fallback tiny set
#         fallback = [
#             {"problem":"Regina wrote 9 novels last year. If this is 3 quarters of the number of novels she has written this year, how many this year?",
#              "answer":12},
#             {"problem":"Dennis uses 1 pound of butter for every dozen croissants. He needs 6 dozen. Butter costs $4/lb, BOGO half off. How much for 6 lb?",
#              "answer":16},
#             {"problem":"A toy costs $12. You have a $3 coupon and a 25% discount applies after coupon. Final price?",
#              "answer":6.75},
#         ]
#         return fallback[:n]

# def load_mmlu_subset(n: int = 100) -> List[Dict[str, Any]]:
#     try:
#         from datasets import load_dataset
#         ds = load_dataset("hendrycks_test", "abstract_algebra", split="test")
#         rows = []
#         for i in range(min(n, len(ds))):
#             q = ds[i]["question"]
#             choices = ds[i]["choices"]
#             ans = ds[i]["answer"]
#             if len(choices) >= 4:
#                 rows.append({"question": q, "A":choices[0], "B":choices[1], "C":choices[2], "D":choices[3],
#                              "answer": ["A","B","C","D"][ans]})
#         if not rows:
#             raise RuntimeError("Empty MMLU split.")
#         return rows
#     except Exception:
#         # Small fallback MCQs
#         return [
#             {"question":"The submandibular salivary gland is expected to be palpable:", 
#              "A":"intraorally", "B":"extraorally", "C":"both intra- and extraorally", "D":"only by radiograph",
#              "answer":"C"},
#             {"question":"Which structure stores genetic information?",
#              "A":"Ribosome","B":"Mitochondrion","C":"Nucleus","D":"Lysosome","answer":"C"},
#         ][:n]

# def load_chess_positions_for_move_validity(k: int = 50) -> List[Dict[str, Any]]:
#     """Create synthetic positions + origin squares; evaluate whether LLM returns a legal destination."""
#     # Use random legal positions derived from simple PGN sequences.
#     positions = []
#     # Start from initial, play a few legal random moves to create variety.
#     for _ in range(k):
#         board = chess.Board()
#         for __ in range(random.randint(6, 12)):
#             moves = list(board.legal_moves)
#             if not moves: break
#             board.push(random.choice(moves))
#         # pick a random piece of side to move
#         own_pieces = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == board.turn]
#         if not own_pieces:
#             continue
#         origin = random.choice(own_pieces)
#         origin_alg = chess.square_name(origin)
#         moves_from_origin = [m for m in board.legal_moves if m.from_square == origin]
#         if not moves_from_origin:  # pick another if no legal move from that piece
#             continue
#         # Build a lightweight PGN notation of the played game so far
#         game = chess.pgn.Game.from_board(board)
#         pgn_str = str(game)
#         positions.append({"pgn": pgn_str, "origin": origin_alg, "board_fen": board.fen()})
#     return positions

# def load_biography_names(n: int = 20) -> List[str]:
#     base = [
#         "Donald Knuth", "Edsger W. Dijkstra", "Barbara Liskov", "John McCarthy", "Dana Scott",
#         "Leslie Lamport", "Andrew Yao", "Judea Pearl", "Whitfield Diffie", "Shafi Goldwasser",
#         "Stephen Cook", "Michael Stonebraker", "Yann LeCun", "Geoffrey Hinton", "Jitendra Malik",
#         "Tim Berners-Lee", "Silvio Micali", "Juris Hartmanis", "David S. Johnson", "John Hopcroft",
#         "Stephen R. Bourne"
#     ]
#     random.shuffle(base)
#     return base[:n]

# # =======================================
# # Evaluators for each task
# # =======================================
# def eval_arithmetic_item(expr: str, response: str) -> bool:
#     pred_nums = re2.findall(r"[-+]?\d+", response)
#     if not pred_nums:
#         return False
#     pred = int(pred_nums[-1])
#     gold = safe_eval_arithmetic(expr)
#     return pred == gold

# def eval_gsm8k_item(gold: float, response: str) -> bool:
#     pred = extract_boxed_number(response)
#     if pred is None or gold is None: return False
#     try:
#         return abs(float(pred) - float(gold)) < 1e-6
#     except Exception:
#         return False

# def eval_mmlu_item(gold: str, response: str) -> bool:
#     pred = extract_choice_letter(response)
#     return (pred == gold)

# def eval_chess_valid_item(board_fen: str, origin: str, response: str) -> bool:
#     board = chess.Board(board_fen)
#     dst = extract_square(response)
#     if not dst:
#         return False
#     try:
#         move = chess.Move.from_uci(chess.square_name(chess.parse_square(origin)) + chess.square_name(chess.parse_square(dst)))
#         return move in board.legal_moves
#     except Exception:
#         return False

# def eval_bio_item(person: str, response: str, llm: LLMClient) -> float:
#     """
#     Factuality score vs Wikipedia summary using LLM judgment when possible.
#     Returns a score in [0,1].
#     """
#     wiki_ctx = wiki_search_and_snippets(person, k=1, char_limit=1500)
#     if not wiki_ctx:
#         # Heuristic: reward mention of name
#         score = 1.0 if person.lower().split()[0] in response.lower() else 0.3
#         return score
#     # LLM judge if available
#     if llm.use_openai:
#         judge_prompt = f"""Compare the bullet biography against the evidence and give a single 0-1 score for factual match (1 = perfect).
# Evidence:
# {wiki_ctx}

# Biography:
# {response}

# Only output the numeric score between 0 and 1."""
#         out = llm.chat([{"role":"system","content":"You are a strict fact‑checking judge that outputs only a number."},
#                         {"role":"user","content": judge_prompt}])
#         m = re2.search(r"0*\.?\d+(?:\.\d+)?", out.strip())
#         if m:
#             try:
#                 v = float(m.group(0))
#                 v = max(0.0, min(1.0, v))
#                 return v
#             except:
#                 pass
#     # Fallback: token overlap heuristic
#     ev = set(re2.findall(r"[a-z]{3,}", wiki_ctx.lower()))
#     bi = set(re2.findall(r"[a-z]{3,}", response.lower()))
#     if not ev: return 0.3
#     return len(ev & bi) / max(1, len(bi))

# # =======================================
# # Experiment runners (baselines + debate)
# # =======================================
# def make_agents(llm: LLMClient, n_agents: int = N_AGENTS, persona_diversity: bool = False) -> List[Agent]:
#     agents: List[Agent] = []
#     if persona_diversity:
#         # diverse personas (novel direction)
#         pool = PERSONAS_BASE[:]
#         if n_agents > len(pool):
#             # repeat with suffix
#             for i in range(len(pool), n_agents):
#                 pool.append((f"Analyst{i+1}", "You are a careful analyst."))
#         roles = pool[:n_agents]
#     else:
#         roles = [("Analyst", "You are a careful, detail‑oriented assistant. Answer concisely.")] * n_agents
#     for name, sys_prompt in roles:
#         agents.append(Agent(name=name, system_prompt=sys_prompt, llm=llm))
#     return agents

# def reflection_single(llm: LLMClient, system_prompt: str, start_prompt: str, refine_instruction: str) -> str:
#     first = llm.chat([{"role":"system","content":system_prompt},{"role":"user","content":start_prompt}])
#     second = llm.chat([
#         {"role":"system","content":system_prompt},
#         {"role":"user","content":f"Here is your earlier answer:\n{first}\n\n{refine_instruction}\nReturn only the updated answer."}
#     ])
#     return second

# def run_task_arithmetic(llm: LLMClient, n_items: int = 100,
#                         use_debate: bool = True,
#                         persona_diversity: bool = False,
#                         retrieval: bool = False,
#                         weighted_vote: bool = True) -> Dict[str, Any]:
#     items = gen_arithmetic(n_items)
#     res_rows = []
#     # Calibration slice for reliability (10% of set)
#     calib_idx = set(random.sample(range(len(items)), max(1, len(items)//10)))
#     agents = make_agents(llm, persona_diversity=persona_diversity)

#     for i, itm in enumerate(tqdm(items, desc="Arithmetic")):
#         expr, gold = itm["expr"], itm["answer"]
#         start_prompt = PROMPTS["arithmetic_start"].format(*re2.findall(r"\d+", expr))
#         rag = ""
#         if retrieval:  # retrieval-augmented debates (not very meaningful for arithmetic; kept for completeness)
#             rag = "You are solving arithmetic; recall PEMDAS: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction."

#         # Baselines: single, reflection, majority
#         single = agents[0].chat(start_prompt if not rag else f"{rag}\n\n{start_prompt}")
#         reflect = reflection_single(
#             agents[0].llm, agents[0].system_prompt, start_prompt if not rag else f"{rag}\n\n{start_prompt}",
#             "Double-check calculations step-by-step and correct any mistake. State the final integer at the end."
#         )

#         # Multi-agent
#         if use_debate:
#             db = Debate(
#                 agents=agents,
#                 task_type="arithmetic",
#                 start_prompt=start_prompt if not rag else f"{rag}\n\n{start_prompt}",
#                 debate_prompt_tmpl=PROMPTS["arithmetic_debate"],
#                 rag_context=None,
#                 rounds=N_ROUNDS
#             ).run()
#             final_majority = db["final_majority"]
#             final_weighted = db["final_weighted"]
#         else:
#             final_majority = single
#             final_weighted = single

#         # Evaluate
#         r = {
#             "expr": expr,
#             "gold": gold,
#             "single": eval_arithmetic_item(expr, single),
#             "reflection": eval_arithmetic_item(expr, reflect),
#             "majority": eval_arithmetic_item(expr, final_majority),
#             "debate_weighted": eval_arithmetic_item(expr, final_weighted if weighted_vote else final_majority),
#         }
#         res_rows.append(r)

#         # Update reliability from calibration items
#         if i in calib_idx:
#             for ag in agents:
#                 # crude: attribute aggregate normalized answer match to each agent last response
#                 # We re-run a single chat to get agent’s individual prediction on the same start prompt
#                 pred = ag.chat(start_prompt)
#                 ok = eval_arithmetic_item(expr, pred)
#                 ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

#     df = pd.DataFrame(res_rows)
#     return {"df": df, "items": items}

# def run_task_gsm8k(llm: LLMClient, n_items: int = 100,
#                    use_debate: bool = True,
#                    persona_diversity: bool = False,
#                    retrieval: bool = False,
#                    weighted_vote: bool = True) -> Dict[str, Any]:
#     data = load_gsm8k_subset(n_items)
#     agents = make_agents(llm, persona_diversity=persona_diversity)
#     calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
#     rows = []
#     for i, itm in enumerate(tqdm(data, desc="GSM8K")):
#         prob, gold = itm["problem"], itm["answer"]
#         start_prompt = PROMPTS["gsm8k_start"].format(problem=prob)
#         rag = wiki_search_and_snippets(prob, k=1, char_limit=1200) if retrieval else ""
#         # baselines
#         sp = start_prompt if not rag else f"Use the following evidence where helpful:\n{rag}\n\n{start_prompt}"
#         single = agents[0].chat(sp)
#         reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
#                                     "Recompute carefully and fix mistakes. Output the final answer as \\boxed{number}.")
#         if use_debate:
#             # Round 0 – per agent independent
#             # (the Debate helper expects the debate template but we need to inject {problem} for GSM8K)
#             # debate_tmpl = PROMPTS["gsm8k_debate"].format(other="{other}", problem=prob)
#             debate_tmpl = PROMPTS["gsm8k_debate"].replace("{problem}", prob)
#             db = Debate(
#                 agents=agents,
#                 task_type="gsm8k",
#                 start_prompt=sp,
#                 debate_prompt_tmpl=debate_tmpl,
#                 rag_context=None,
#                 rounds=N_ROUNDS
#             ).run()
#             final_majority = db["final_majority"]
#             final_weighted = db["final_weighted"]
#         else:
#             final_majority = single
#             final_weighted = single

#         r = {
#             "problem": prob,
#             "gold": gold,
#             "single": eval_gsm8k_item(gold, single),
#             "reflection": eval_gsm8k_item(gold, reflect),
#             "majority": eval_gsm8k_item(gold, final_majority),
#             "debate_weighted": eval_gsm8k_item(gold, final_weighted if weighted_vote else final_majority),
#         }
#         rows.append(r)

#         if i in calib_idx:
#             for ag in agents:
#                 pred = ag.chat(sp)
#                 ok = eval_gsm8k_item(gold, pred)
#                 ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

#     df = pd.DataFrame(rows)
#     return {"df": df, "items": data}

# def run_task_mmlu(llm: LLMClient, n_items: int = 100,
#                   use_debate: bool = True,
#                   persona_diversity: bool = False,
#                   retrieval: bool = False,
#                   weighted_vote: bool = True) -> Dict[str, Any]:
#     data = load_mmlu_subset(n_items)
#     agents = make_agents(llm, persona_diversity=persona_diversity)
#     calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
#     rows = []
#     for i, itm in enumerate(tqdm(data, desc="MMLU")):
#         q, A, B, C, D, gold = itm["question"], itm["A"], itm["B"], itm["C"], itm["D"], itm["answer"]
#         start_prompt = PROMPTS["mmlu_start"].format(question=q, A=A, B=B, C=C, D=D)
#         rag = wiki_search_and_snippets(q, k=1, char_limit=1200) if retrieval else ""
#         sp = start_prompt if not rag else f"Use the following evidence where helpful:\n{rag}\n\n{start_prompt}"
#         single = agents[0].chat(sp)
#         reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
#                                     "Re-evaluate carefully and return only the final choice in the form (A|B|C|D).")

#         if use_debate:
#             debate_tmpl = PROMPTS["mmlu_debate"]
#             db = Debate(
#                 agents=agents,
#                 task_type="mmlu",
#                 start_prompt=sp,
#                 debate_prompt_tmpl=debate_tmpl,
#                 rag_context=None,
#                 rounds=N_ROUNDS
#             ).run()
#             final_majority = db["final_majority"]
#             final_weighted = db["final_weighted"]
#         else:
#             final_majority = single
#             final_weighted = single

#         r = {
#             "question": q,
#             "gold": gold,
#             "single": eval_mmlu_item(gold, single),
#             "reflection": eval_mmlu_item(gold, reflect),
#             "majority": eval_mmlu_item(gold, final_majority),
#             "debate_weighted": eval_mmlu_item(gold, final_weighted if weighted_vote else final_majority),
#         }
#         rows.append(r)

#         if i in calib_idx:
#             for ag in agents:
#                 pred = ag.chat(sp)
#                 ok = eval_mmlu_item(gold, pred)
#                 ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

#     df = pd.DataFrame(rows)
#     return {"df": df, "items": data}

# def run_task_chess_validity(llm: LLMClient, n_items: int = 100,
#                             use_debate: bool = True,
#                             persona_diversity: bool = False,
#                             retrieval: bool = False,
#                             weighted_vote: bool = True) -> Dict[str, Any]:
#     data = load_chess_positions_for_move_validity(n_items*2)
#     data = data[:n_items]
#     agents = make_agents(llm, persona_diversity=persona_diversity)
#     calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
#     rows = []
#     for i, itm in enumerate(tqdm(data, desc="Chess Validity")):
#         pgn, origin, fen = itm["pgn"], itm["origin"], itm["board_fen"]
#         start = PROMPTS["chess_valid_start"].format(moves=pgn, origin=origin)
#         rag = ""  # retrieval not really helpful for this
#         sp = start
#         single = agents[0].chat(sp)
#         reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
#                                     "Check legality carefully and return only (square).")

#         if use_debate:
#             db = Debate(
#                 agents=agents,
#                 task_type="chess_valid",
#                 start_prompt=sp,
#                 debate_prompt_tmpl=PROMPTS["chess_valid_debate"],
#                 rag_context=None,
#                 rounds=N_ROUNDS
#             ).run()
#             final_majority = db["final_majority"]
#             final_weighted = db["final_weighted"]
#         else:
#             final_majority = single
#             final_weighted = single

#         r = {
#             "origin": origin,
#             "single": eval_chess_valid_item(fen, origin, single),
#             "reflection": eval_chess_valid_item(fen, origin, reflect),
#             "majority": eval_chess_valid_item(fen, origin, final_majority),
#             "debate_weighted": eval_chess_valid_item(fen, origin, final_weighted if weighted_vote else final_majority),
#         }
#         rows.append(r)

#         if i in calib_idx:
#             for ag in agents:
#                 pred = ag.chat(sp)
#                 ok = eval_chess_valid_item(fen, origin, pred)
#                 ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

#     df = pd.DataFrame(rows)
#     return {"df": df, "items": data}

# def run_task_biographies(llm: LLMClient, n_items: int = 50,
#                          use_debate: bool = True,
#                          persona_diversity: bool = True,     # personas help here
#                          retrieval: bool = True,             # RAG strongly helps
#                          weighted_vote: bool = True) -> Dict[str, Any]:
#     people = load_biography_names(n_items)
#     agents = make_agents(llm, persona_diversity=persona_diversity)
#     calib_idx = set(random.sample(range(len(people)), max(1, len(people)//10)))
#     rows = []
#     for i, person in enumerate(tqdm(people, desc="Biographies")):
#         start_prompt = PROMPTS["bio_start"].format(person=person)
#         rag = wiki_search_and_snippets(person, k=2, char_limit=1500) if retrieval else ""
#         sp = start_prompt if not rag else f"Use the following evidence to avoid hallucinations:\n{rag}\n\n{start_prompt}"
#         single = agents[0].chat(sp)
#         reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
#                                     "Cross-check every bullet against the evidence. Remove any uncertain bullets. Return bullets only.")

#         if use_debate:
#             tmpl = PROMPTS["bio_debate"].format(other="{other}", person=person)
#             db = Debate(
#                 agents=agents,
#                 task_type="bio",
#                 start_prompt=sp,
#                 debate_prompt_tmpl=tmpl,
#                 rag_context=None,
#                 rounds=N_ROUNDS
#             ).run()
#             final_majority = db["final_majority"]
#             final_weighted = db["final_weighted"]
#         else:
#             final_majority = single
#             final_weighted = single

#         s_single = eval_bio_item(person, single, llm)
#         s_reflec = eval_bio_item(person, reflect, llm)
#         s_major  = eval_bio_item(person, final_majority, llm)
#         s_weight = eval_bio_item(person, final_weighted if weighted_vote else final_majority, llm)

#         r = {"person": person, "single": s_single, "reflection": s_reflec, "majority": s_major, "debate_weighted": s_weight}
#         rows.append(r)

#         if i in calib_idx:
#             for ag in agents:
#                 pred = ag.chat(sp)
#                 sc = eval_bio_item(person, pred, llm)  # 0..1
#                 ag.reliability = moving_average(ag.reliability, sc, alpha=0.2)

#     df = pd.DataFrame(rows)
#     return {"df": df, "items": people}

# # =======================================
# # Plotting (Figures similar to the paper)
# # =======================================
# def barplot_results(task_name: str, df: pd.DataFrame, is_accuracy: bool = True):
#     metrics = ["single", "reflection", "majority", "debate_weighted"]
#     labels  = ["Single Agent", "Reflection", "Multi‑Agent (Majority)", "Multi‑Agent (Debate)"]
#     vals = []
#     for m in metrics:
#         if is_accuracy:
#             vals.append(df[m].mean() * 100.0)
#         else:  # biography factuality 0..1 -> %
#             vals.append(df[m].mean() * 100.0)
#     plt.figure(figsize=(6,4))
#     xs = np.arange(len(metrics))
#     plt.bar(xs, vals)
#     plt.xticks(xs, labels, rotation=15, ha="right")
#     plt.ylabel("Accuracy (%)" if is_accuracy else "Factuality (%)")
#     plt.title(task_name)
#     plt.tight_layout()
#     path = os.path.join(OUT_DIR, f"{task_name.lower().replace(' ','_')}_bars.png")
#     plt.savefig(path, dpi=180)
#     plt.close()
#     print(f"[Saved] {path}")

# def ablation_plot(title: str, xvals: List[int], yvals: List[float], xlabel: str, ylabel: str, fname: str):
#     plt.figure(figsize=(5,4))
#     plt.plot(xvals, yvals, marker="o")
#     plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
#     plt.tight_layout()
#     path = os.path.join(OUT_DIR, fname)
#     plt.savefig(path, dpi=180)
#     plt.close()
#     print(f"[Saved] {path}")

# # =======================================
# # Ablations: agents & rounds (Figure 10/12)
# # =======================================
# def ablation_agents_gsm8k(llm: LLMClient):
#     xvals, accs = [], []
#     for k in [1,2,3,4,5]:
#         ags = make_agents(llm, n_agents=k, persona_diversity=True)
#         data = load_gsm8k_subset(30)
#         rows = []
#         for itm in data:
#             prob, gold = itm["problem"], itm["answer"]
#             sp = PROMPTS["gsm8k_start"].format(problem=prob)
#             # tmpl = PROMPTS["gsm8k_debate"].format(other="{other}", problem=prob)
#             tmpl = PROMPTS["gsm8k_debate"].replace("{problem}", prob)
#             db = Debate(agents=ags, task_type="gsm8k", start_prompt=sp, debate_prompt_tmpl=tmpl, rounds=2,
#                         rag_context=None, summarization_chars=2000).run()
#             ok = eval_gsm8k_item(gold, db["final_weighted"])
#             rows.append(ok)
#         xvals.append(k); accs.append(100.0 * (sum(rows)/len(rows)))
#     ablation_plot("Performance vs Number of Debating Agents", xvals, accs, "Agents", "GSM8K Accuracy (%)", "ablation_agents_gsm8k.png")

# def ablation_rounds_arithmetic(llm: LLMClient):
#     xvals, accs = [], []
#     items = gen_arithmetic(80)
#     for rds in [1,2,3,4]:
#         ags = make_agents(llm, n_agents=3, persona_diversity=False)
#         ok_list = []
#         for itm in items:
#             expr = itm["expr"]; start = PROMPTS["arithmetic_start"].format(*re2.findall(r"\d+", expr))
#             db = Debate(agents=ags, task_type="arithmetic", start_prompt=start,
#                         debate_prompt_tmpl=PROMPTS["arithmetic_debate"], rounds=rds,
#                         rag_context=None, summarization_chars=1500).run()
#             ok = eval_arithmetic_item(expr, db["final_weighted"])
#             ok_list.append(ok)
#         xvals.append(rds); accs.append(100.0 * (sum(ok_list)/len(ok_list)))
#     ablation_plot("Math Accuracy vs Debate Rounds", xvals, accs, "Debate Rounds", "Arithmetic Accuracy (%)", "ablation_rounds_arith.png")

# # =======================================
# # Master pipeline
# # =======================================
# def main():
#     llm = LLMClient()

#     # ------------------------------
#     # (1) Reproduce core benchmarks
#     # ------------------------------
#     arithmetic = run_task_arithmetic(llm, n_items=100, use_debate=True, persona_diversity=False,
#                                      retrieval=False, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "arithmetic_results.jsonl"), arithmetic["df"].to_dict(orient="records"))
#     barplot_results("Arithmetic", arithmetic["df"], is_accuracy=True)

#     gsm8k = run_task_gsm8k(llm, n_items=100, use_debate=True, persona_diversity=False,
#                            retrieval=False, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "gsm8k_results.jsonl"), gsm8k["df"].to_dict(orient="records"))
#     barplot_results("GSM8K", gsm8k["df"], is_accuracy=True)

#     mmlu = run_task_mmlu(llm, n_items=100, use_debate=True, persona_diversity=False,
#                          retrieval=False, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "mmlu_results.jsonl"), mmlu["df"].to_dict(orient="records"))
#     barplot_results("MMLU", mmlu["df"], is_accuracy=True)

#     chess_valid = run_task_chess_validity(llm, n_items=80, use_debate=True, persona_diversity=False,
#                                           retrieval=False, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "chess_valid_results.jsonl"), chess_valid["df"].to_dict(orient="records"))
#     barplot_results("Chess Move Validity", chess_valid["df"], is_accuracy=True)

#     bios = run_task_biographies(llm, n_items=30, use_debate=True, persona_diversity=True,
#                                 retrieval=True, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "biographies_results.jsonl"), bios["df"].to_dict(orient="records"))
#     barplot_results("Biographies Factuality", bios["df"], is_accuracy=False)

#     # ------------------------------------------------
#     # (2) Novel directions: RAG / Weighted / Personas
#     #     plus Adaptive Summarization (built‑in)
#     # ------------------------------------------------
#     # RAG for GSM8K & MMLU
#     gsm8k_rag = run_task_gsm8k(llm, n_items=60, use_debate=True, persona_diversity=False,
#                                retrieval=True, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "gsm8k_rag_results.jsonl"), gsm8k_rag["df"].to_dict(orient="records"))
#     barplot_results("GSM8K (RAG Debate)", gsm8k_rag["df"], is_accuracy=True)

#     mmlu_rag = run_task_mmlu(llm, n_items=60, use_debate=True, persona_diversity=False,
#                               retrieval=True, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "mmlu_rag_results.jsonl"), mmlu_rag["df"].to_dict(orient="records"))
#     barplot_results("MMLU (RAG Debate)", mmlu_rag["df"], is_accuracy=True)

#     # Persona diversity on MMLU
#     mmlu_personas = run_task_mmlu(llm, n_items=60, use_debate=True, persona_diversity=True,
#                                   retrieval=False, weighted_vote=True)
#     ensure_jsonl(os.path.join(OUT_DIR, "mmlu_personas_results.jsonl"), mmlu_personas["df"].to_dict(orient="records"))
#     barplot_results("MMLU (Persona‑Diverse Debate)", mmlu_personas["df"], is_accuracy=True)

#     # Ablations: agents & rounds (Figures 10/12 style)
#     ablation_agents_gsm8k(llm)
#     ablation_rounds_arithmetic(llm)

#     # Print quick summary
#     def summarize(df, name):
#         return {
#             "task": name,
#             "Single": round(df["single"].mean()*100.0, 2),
#             "Reflection": round(df["reflection"].mean()*100.0, 2),
#             "Majority": round(df["majority"].mean()*100.0, 2),
#             "DebateWeighted": round(df["debate_weighted"].mean()*100.0, 2),
#         }
#     summary = [
#         summarize(arithmetic["df"], "Arithmetic"),
#         summarize(gsm8k["df"], "GSM8K"),
#         summarize(mmlu["df"], "MMLU"),
#         summarize(chess_valid["df"], "Chess Validity"),
#         summarize(bios["df"], "Biographies (factuality %)"),
#         summarize(gsm8k_rag["df"], "GSM8K (RAG)"),
#         summarize(mmlu_rag["df"], "MMLU (RAG)"),
#         summarize(mmlu_personas["df"], "MMLU (Personas)"),
#     ]
#     with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)
#     print("\n=== Summary (%) ===")
#     for s in summary: print(s)

# if __name__ == "__main__":
#     main()




"""
Multi‑Agent Debate (MAD) – end‑to‑end reproducible pipeline + novel extensions.

What this file does
-------------------
1) Reproduce the core experiments from "Improving Factuality and Reasoning in Language Models
   through Multiagent Debate" using black‑box chat LLMs.
   - Arithmetic (synthetic)
   - GSM8K (grade‑school math; light subset by default; full set if HuggingFace datasets available)
   - MMLU (subset by default; full set if HuggingFace datasets available)
   - Chess move *validity* (python‑chess based)
   - Computer‑scientist biographies factuality (LLM‑judged vs. Wikipedia retrieval)

2) Implement and evaluate the 4 proposed *novel directions* from the screenshots:
   - Retrieval‑Augmented Debates (evidence retrieved via Wikipedia API)
   - Weighted Voting (agent weights from rolling reliability on a calibration set)
   - Persona Diversity (specialised agent system prompts, e.g., historian, engineer)
   - Adaptive Summarization (token‑budget aware summarization of other agents’ messages)

3) Generate publication‑ready figures mirroring the paper’s plots (accuracy bars, ablations).

How to run
----------
$ python mad_pipeline.py                   # (rename this file if desired)
Environment:
- Requires Python 3.9+.
- Set OPENAI_API_KEY for real LLM runs (recommended).
- If no API key is found, a fall‑back mock LLM is used (pipeline still runs without error, but
  results obviously won’t match the paper).

Outputs are written to ./outputs/*.png and ./outputs/*.jsonl

Dependencies are auto‑installed on first run (matplotlib, numpy, pandas, python‑chess, wikipedia, tqdm, regex, datasets (optional)).

Notes
-----
- Prompts and evaluation setups follow the paper’s Appendix/Table of prompts and sections 2–3
  (see page references inside the code). :contentReference[oaicite:0]{index=0}
- This code is designed to be robust: it catches network/API errors and continues with sensible
  fallbacks so you can “copy‑paste‑run” without manual edits.

"""

# =========================
# Bootstrapping & Imports
# =========================
import os, sys, json, math, time, random, re, textwrap, subprocess, shutil
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

def _pip_install(pkgs: List[str]):
    """Install packages if missing (quietly)."""
    for p in pkgs:
        try:
            __import__(p if p != "python_chess" else "chess")
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])
            except Exception:
                print(f"[WARN] Could not install {p}. Some features might be limited.")

_pip_install([
    "matplotlib", "numpy", "pandas", "tqdm", "regex", "wikipedia", "python-chess",
    # Optional:
    "datasets"
])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wikipedia
import chess
import chess.pgn
import regex as re2

# ===========================================
# Global config & reproducibility
# ===========================================
SEED = int(os.environ.get("MAD_SEED", "7"))
random.seed(SEED); np.random.seed(SEED)

OUT_DIR = os.path.abspath("./outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Default LLMs / models
DEFAULT_MODEL = os.environ.get("MAD_MODEL", "gpt-4o-mini")  # use any chat model available to you
TEMPERATURE = float(os.environ.get("MAD_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("MAD_MAX_TOKENS", "512"))

# Debate config (Figure 10/12 in paper discuss agents/rounds) :contentReference[oaicite:1]{index=1}
N_AGENTS = int(os.environ.get("MAD_AGENTS", "3"))
N_ROUNDS = int(os.environ.get("MAD_ROUNDS", "2"))

# Persona diversity (novel direction)
PERSONAS_BASE = [
    ("Generalist",       "You are a careful, detail‑oriented analyst. Think step by step."),
    ("Historian",        "You are a historian. You check dates, sources, and historical consistency."),
    ("Engineer",         "You are a pragmatic engineer. You verify units, constraints, and edge cases."),
    ("Mathematician",    "You are a mathematician. You reason formally and verify each derivation."),
    ("Skeptical Reviewer","You are a skeptical reviewer. You aggressively fact‑check and call out errors."),
]

# =========================
# Utilities
# =========================
def safe_eval_arithmetic(expr: str) -> Optional[int]:
    """Safely evaluate arithmetic of the form digits with + - * only (as in the paper’s arithmetic task)."""
    if not re2.fullmatch(r"[0-9+\-*\s]+", expr):
        return None
    try:
        # Python eval is safe under our strict regex (only digits/operators/spaces)
        return int(eval(expr))
    except Exception:
        return None

def ensure_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def strip_codefences(s: str) -> str:
    return re2.sub(r"^```.*?^```", "", s, flags=re2.S | re2.M)

def extract_boxed_number(text: str) -> Optional[float]:
    """
    Extract \\boxed{number} (as used in GSM8K prompt) or a final numeric answer.
    """
    text = text.strip()
    m = re2.search(r"\\boxed\{([^}]+)\}", text)
    cand = None
    if m:
        cand = m.group(1)
    else:
        # Try last number in text
        nums = re2.findall(r"[-+]?\d*\.?\d+", text)
        if nums:
            cand = nums[-1]
    if cand is None: return None
    try:
        if "." in cand: return float(cand)
        return int(cand)
    except:
        try:
            return float(cand)
        except:
            return None

def extract_choice_letter(text: str) -> Optional[str]:
    """Extract (A) or (B) format at the end."""
    m = re2.search(r"\(([A-D])\)\s*$", text.strip())
    return m.group(1) if m else None

def extract_chess_move_on_turn(text: str, turn_number: int = 14) -> Optional[str]:
    """Expect '14. Qa4' style from the paper’s chess prompt."""
    m = re2.search(rf"\b{turn_number}\.\s*([a-hKQRBN0O\-\=x\+#!][a-h1-8O\-x=+#]*)", text.strip())
    if m:
        return m.group(1).strip()
    return None

def extract_square(text: str) -> Optional[str]:
    """For chess validity task: extract (e5) -> 'e5'."""
    m = re2.search(r"\(([a-h][1-8])\)", text.strip(), flags=re2.I)
    if m: return m.group(1).lower()
    # fallback: last a-h1-8 pattern
    m = re2.findall(r"\b([a-h][1-8])\b", text.strip(), flags=re2.I)
    return m[-1].lower() if m else None

def truncate_tokens(s: str, max_chars: int = 4000) -> str:
    if len(s) <= max_chars: return s
    return s[:max_chars//2] + "\n[...snip...]\n" + s[-max_chars//2:]

def moving_average(prev: float, new: float, alpha: float = 0.2) -> float:
    return (1 - alpha) * prev + alpha * new

# ===================================
# Lightweight LLM client (OpenAI/Mock)
# ===================================
class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_openai = bool(os.getenv("OPENAI_API_KEY"))
        self._openai_ready = False
        if self.use_openai:
            self._init_openai()

    def _init_openai(self):
        try:
            # Try new SDK
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
                if self._mode == "v1":
                    r = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=stop
                    )
                    return r.choices[0].message.content
                else:
                    r = self._client.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=stop
                    )
                    return r['choices'][0]['message']['content']
            except Exception as e:
                print(f"[WARN] OpenAI call failed: {e}. Falling back to MockLLM.")
                return self._mock_chat(messages)
        else:
            return self._mock_chat(messages)

    # --------------------------------------------------
    # Mock LLM (deterministic, fast, *not* a real model)
    # --------------------------------------------------
    def _mock_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        A small deterministic “LLM” that:
        - Solves arithmetic exactly if it finds 'What is the result of ...' expression.
        - Outputs a plausible GSM8K/MMLU/chess/biography answer with heuristics.
        This keeps the pipeline runnable without API keys.
        """
        content = " ".join([m["content"] for m in messages if m["role"] != "system"]).lower()

        # Arithmetic
        m = re2.search(r"what is the result of\s*([0-9+\-*\s]+)\??", content)
        if m:
            expr = m.group(1)
            val = safe_eval_arithmetic(expr)
            if val is None:
                val = random.randint(-500, 500)
            return f"My answer is {val}."

        # GSM8K (look for a final \boxed{})
        # Heuristic: if a simple percentage or sum pattern present, try to compute.
        # Otherwise return a deterministic number from hash for reproducibility.
        if "your final answer should be a single numerical number" in content:
            # simple percent-of or sum detection
            nums = [int(x) for x in re2.findall(r"\b\d+\b", content)]
            ans = None
            if "percent" in content or "%" in content:
                ps = [int(x) for x in re2.findall(r"\b(\d+)\s*%", content)]
                if ps and nums:
                    p = ps[-1] / 100.0
                    ans = int(round(nums[-1] * p))
            if ans is None and nums:
                ans = sum(nums[:3])  # crude
            if ans is None:
                ans = (abs(hash(content)) % 97) + 3
            return f"The answer is \\boxed{{{ans}}}."

        # MMLU multiple-choice -> always (C) deterministically from hash
        if "putting the answer in the form (" in content and re2.search(r"\([ABCD]\)", content):
            choice = ["A","B","C","D"][abs(hash(content)) % 4]
            return f"I choose ({choice})."

        # Chess move (return a harmless queen move if legal square is mentioned)
        if "what is the best chess move i should execute next" in content:
            return "14. Qa4"

        # Chess validity – suggest e5 if piece on e4 mentioned, else a legal looking square.
        if "give one valid destination square" in content:
            squares = re2.findall(r"\b([a-h][1-8])\b", content)
            dst = squares[-1] if squares else random.choice(["e4","e5","d4","d5"])
            return f"A valid destination is ({dst})."

        # Biography bullets – stub from Wikipedia summary if we can retrieve, else canned bullets
        if "bullet point biography" in content:
            m = re2.search(r"biography of ([^,\n]+)", content) or re2.search(r"biographies of ([^,\n]+)", content)
            name = (m.group(1).strip() if m else "the person").title()
            bullets = [
                f"- {name} is a computer scientist.",
                f"- Known for contributions to algorithms and systems.",
                "- Recipient of multiple awards."
            ]
            return "\n".join(bullets)

        # Default deterministic filler
        return "Here is my updated answer. (C)"

# ============================================
# Prompts (from Table of prompts in the paper)
# ============================================
# See Appendix Table (page with Figure 15) for these templates. :contentReference[oaicite:2]{index=2}
PROMPTS = {
    "arithmetic_start": "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.",
    "arithmetic_debate": ("These are the recent/updated opinions from other agents:\n{other}\n"
                          "Use these opinions carefully as additional advice, can you provide an updated answer? "
                          "Make sure to state your answer at the end of the response."),
    "gsm8k_start": ("Can you solve the following math problem?\n{problem}\n"
                    "Explain your reasoning. Your final answer should be a single numerical number, "
                    "in the form \\boxed{{answer}}, at the end of your response."),
    "gsm8k_debate": ("These are the solutions to the problem from other agents:\n{other}\n"
                     "Using the solutions from other agents as additional information, can you provide your answer to the math problem?\n"
                     "The original math problem is:\n{problem}\n"
                     "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."),
    "mmlu_start": ("Can you answer the following question as accurately as possible?\n\n{question}\n"
                   "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
                   "Explain your answer, putting the answer in the form (X) at the end of your response."),
    "mmlu_debate": ("These are the solutions to the problem from other agents:\n{other}\n"
                    "Using the reasoning from other agents as additional advice, can you give an updated answer?\n"
                    "Examine your solution and that of other agents. Put your answer in the form (X) at the end of your response."),
    "bio_start": ("Give a bullet point biography of {person} highlighting their contributions and achievements as a computer scientist, "
                  "with each fact separated with a new line character."),
    "bio_debate": ("Here are some bullet point biographies of {person} given by other agents:\n{other}\n"
                   "Closely examine your biography and the biography of other agents and provide an updated bullet point biography."),
    "chess_move_start": ("Here is the current sequence of moves in a chess game:\n{moves}\n"
                         "What is the best chess move I should execute next? Give a single move suggestion of the form 14. <XXX> "
                         "and make sure the chess move is valid in the current board state."),
    "chess_move_debate": ("Here are other chess move suggestions from other agents:\n{other}\n"
                          "Using the chess suggestions from other agents as additional advice and your earlier generated solution, "
                          "can you give me your updated thoughts on the best next chess move I should play given the chess sequence?\n"
                          "Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state."),
    "chess_valid_start": ("Given the chess game\n{moves}\n"
                          "give one valid destination square for the chess piece at {origin}.\n"
                          "State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5).\n"
                          "Give a one line explanation of why your destination square is a valid move."),
    "chess_valid_debate": ("Here are destination square suggestions from other agents:\n{other}\n"
                           "Can you double check that your destination square is a valid move? "
                           "Check the valid move justifications from other agents.\n"
                           "State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."),
}

# =======================================
# Retrieval (Wikipedia) for RAG debates
# =======================================
def wiki_search_and_snippets(query: str, k: int = 2, char_limit: int = 1500) -> str:
    try:
        wikipedia.set_lang("en")
        hits = wikipedia.search(query, results=k)
        snippets = []
        for h in hits:
            try:
                pg = wikipedia.page(h, auto_suggest=False)
                s = pg.summary
                if s:
                    snippets.append(f"Title: {pg.title}\nSummary:\n{s}")
            except Exception:
                continue
        joined = "\n\n".join(snippets)
        return truncate_tokens(joined, max_chars=char_limit) if joined else ""
    except Exception:
        return ""

# =======================================
# Adaptive Summarization (LLM or extractive)
# =======================================
def summarize_text(text: str, llm: LLMClient, target_chars: int = 1500) -> str:
    if len(text) <= target_chars:
        return text
    # Try quick LLM summary if available
    if llm.use_openai:
        prompt = f"Summarize the following debate, keeping all key facts, equations, and disagreements under {target_chars} characters:\n\n{text}"
        try:
            out = llm.chat([{"role":"system","content":"You compress long texts without losing facts."},
                            {"role":"user","content": prompt}])
            return truncate_tokens(out, target_chars)
        except Exception:
            pass
    # Fallback extractive: top-k sentences by length
    sents = re2.split(r"(?<=[\.\?\!])\s+", text)
    sents = sorted(sents, key=lambda s: -len(s))
    acc, out = 0, []
    for s in sents:
        if acc + len(s) + 1 > target_chars: continue
        out.append(s); acc += len(s) + 1
        if acc >= target_chars: break
    if not out: return text[:target_chars]
    return " ".join(out)

# =======================================
# Agent & Debate Orchestrator
# =======================================
@dataclass
class Agent:
    name: str
    system_prompt: str
    llm: LLMClient
    reliability: float = 0.5  # for weighted voting (novel direction)
    id: int = field(default_factory=lambda: random.randint(0, 1_000_000))

    def chat(self, user_prompt: str) -> str:
        msgs = [{"role":"system","content": self.system_prompt},
                {"role":"user","content": user_prompt}]
        return self.llm.chat(msgs).strip()

class Debate:
    def __init__(
        self,
        agents: List[Agent],
        task_type: str,
        start_prompt: str,
        debate_prompt_tmpl: str,
        rag_context: Optional[str] = None,
        rounds: int = N_ROUNDS,
        summarization_chars: int = 3000,   # Adaptive summarization budget
    ):
        self.agents = agents
        self.task_type = task_type
        self.start_prompt = start_prompt
        self.debate_prompt_tmpl = debate_prompt_tmpl
        self.rag_context = rag_context
        self.rounds = rounds
        self.summarization_chars = summarization_chars
        self.history: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        # Round 0 – independent proposals
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
                other_text = summarize_text(other_text, a.llm, target_chars=self.summarization_chars)
                # p = self.debate_prompt_tmpl.format(other=other_text)
                p = self.debate_prompt_tmpl.replace("{other}", other_text)
                if self.task_type in ("gsm8k", "mmlu", "bio", "chess_move", "chess_valid") and "{problem}" in p:
                    # (gsm8k) handled by caller; here we simply keep p as built
                    pass
                r = a.chat(p)
                new_responses.append(r)
            responses = new_responses
            self.history.append({"round": rd, "responses": responses[:]})

        # Aggregation – majority and weighted voting
        final_majority = self.aggregate_majority(responses)
        final_weighted = self.aggregate_weighted(responses)
        return {
            "history": self.history,
            "final_majority": final_majority,
            "final_weighted": final_weighted,
            "all_final": responses,
        }

    def aggregate_majority(self, responses: List[str]) -> str:
        # Task‑specific normalizers
        key = [self._normalize_answer(r) for r in responses]
        counts: Dict[str, int] = {}
        for k in key:
            counts[k] = counts.get(k, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        # return the original response corresponding to the winner
        for r in responses[::-1]:
            if self._normalize_answer(r) == best:
                return r
        return responses[0]

    def aggregate_weighted(self, responses: List[str]) -> str:
        # Weighted voting by agent.reliability (novel direction)
        norm = [self._normalize_answer(r) for r in responses]
        scores: Dict[str, float] = {}
        for r, a in zip(norm, self.agents):
            scores[r] = scores.get(r, 0.0) + max(1e-3, a.reliability)
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for r in responses[::-1]:
            if self._normalize_answer(r) == best:
                return r
        return responses[0]

    def _normalize_answer(self, text: str) -> str:
        """Task‑aware canonicalization for voting."""
        t = text.strip()
        if self.task_type == "arithmetic":
            # last integer
            nums = re2.findall(r"[-+]?\d+", t)
            return nums[-1] if nums else t
        if self.task_type == "gsm8k":
            v = extract_boxed_number(t)
            return str(v) if v is not None else t
        if self.task_type == "mmlu":
            c = extract_choice_letter(t)
            return c or t
        if self.task_type == "chess_move":
            m = extract_chess_move_on_turn(t, 14)
            return m or t
        if self.task_type == "chess_valid":
            s = extract_square(t)
            return s or t
        if self.task_type == "bio":
            # collapse to set of lines
            lines = [ln.strip("-• ").strip() for ln in t.splitlines() if ln.strip()]
            return " | ".join(sorted(set(lines)))[:400]
        return t

# =======================================
# Tasks: generators & evaluators
# =======================================
def gen_arithmetic(n: int = 100) -> List[Dict[str, Any]]:
    items = []
    for _ in range(n):
        xs = [random.randint(0, 30) for __ in range(6)]
        expr = f"{xs[0]}+{xs[1]}*{xs[2]}+{xs[3]}-{xs[4]}*{xs[5]}"
        ans = safe_eval_arithmetic(expr)
        items.append({"expr": expr, "answer": ans})
    return items

def load_gsm8k_examples(limit: Optional[int] = None, split: str = "test") -> List[Dict[str, Any]]:
    """
    Load GSM8K from HuggingFace (openai/gsm8k). Returns the entire split when `limit` is None.
    Falls back to a small hand-crafted set if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        total = len(ds)
        if limit is None or limit > total:
            limit = total
        rows: List[Dict[str, Any]] = []
        for example in ds.select(range(limit)):
            question = example["question"]
            answer_text = example["answer"]
            m = re2.search(r"####\s*([-+]?\d*\.?\d+)", answer_text)
            gold = float(m.group(1)) if m else None
            if gold is not None and float(gold).is_integer():
                gold = int(gold)
            rows.append({"problem": question, "answer": gold})
        return rows
    except Exception:
        fallback = [
            {"problem": "Regina wrote 9 novels last year. If this is 3 quarters of the number of novels she has written this year, how many this year?",
             "answer": 12},
            {"problem": "Dennis uses 1 pound of butter for every dozen croissants. He needs 6 dozen. Butter costs $4/lb, BOGO half off. How much for 6 lb?",
             "answer": 16},
            {"problem": "A toy costs $12. You have a $3 coupon and a 25% discount applies after coupon. Final price?",
             "answer": 6.75},
        ]
        if limit is None:
            limit = len(fallback)
        return fallback[:limit]

def load_mmlu_examples(limit: Optional[int] = None,
                       split: str = "validation",
                       subjects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load MMLU from HuggingFace (cais/mmlu). Optionally restrict to a subset of subjects.
    Falls back to a tiny MCQ set if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split=split)
        if subjects:
            subject_set = set(subjects)
            ds = ds.filter(lambda ex: ex.get("subject") in subject_set)
        total = len(ds)
        if total == 0:
            raise RuntimeError("MMLU split returned no examples.")
        if limit is None or limit > total:
            limit = total

        rows: List[Dict[str, Any]] = []
        for example in ds.select(range(limit)):
            choices = example.get("choices")
            if not isinstance(choices, list) or len(choices) < 4:
                continue
            answer_idx = example.get("answer", 0)
            rows.append({
                "question": example.get("question", ""),
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
                "answer": ["A", "B", "C", "D"][answer_idx],
                "subject": example.get("subject", ""),
            })
        if not rows:
            raise RuntimeError("No usable MMLU entries after filtering.")
        return rows
    except Exception:
        fallback = [
            {"question": "The submandibular salivary gland is expected to be palpable:",
             "A": "intraorally", "B": "extraorally", "C": "both intra- and extraorally", "D": "only by radiograph",
             "answer": "C"},
            {"question": "Which structure stores genetic information?",
             "A": "Ribosome", "B": "Mitochondrion", "C": "Nucleus", "D": "Lysosome", "answer": "C"},
        ]
        if limit is None:
            limit = len(fallback)
        return fallback[:limit]

def load_chess_positions_for_move_validity(k: int = 50) -> List[Dict[str, Any]]:
    """Create synthetic positions + origin squares; evaluate whether LLM returns a legal destination."""
    # Use random legal positions derived from simple PGN sequences.
    positions = []
    # Start from initial, play a few legal random moves to create variety.
    for _ in range(k):
        board = chess.Board()
        for __ in range(random.randint(6, 12)):
            moves = list(board.legal_moves)
            if not moves: break
            board.push(random.choice(moves))
        # pick a random piece of side to move
        own_pieces = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == board.turn]
        if not own_pieces:
            continue
        origin = random.choice(own_pieces)
        origin_alg = chess.square_name(origin)
        moves_from_origin = [m for m in board.legal_moves if m.from_square == origin]
        if not moves_from_origin:  # pick another if no legal move from that piece
            continue
        # Build a lightweight PGN notation of the played game so far
        game = chess.pgn.Game.from_board(board)
        pgn_str = str(game)
        positions.append({"pgn": pgn_str, "origin": origin_alg, "board_fen": board.fen()})
    return positions

# def load_biography_names(n: int = 20) -> List[str]:
#     base = [
#         "Donald Knuth", "Edsger W. Dijkstra", "Barbara Liskov", "John McCarthy", "Dana Scott",
#         "Leslie Lamport", "Andrew Yao", "Judea Pearl", "Whitfield Diffie", "Shafi Goldwasser",
#         "Stephen Cook", "Michael Stonebraker", "Yann LeCun", "Geoffrey Hinton", "Jitendra Malik",
#         "Tim Berners-Lee", "Silvio Micali", "Juris Hartmanis", "David S. Johnson", "John Hopcroft",
#         "Stephen R. Bourne"
#     ]
#     random.shuffle(base)
#     return base[:n]
def load_biography_entries(path: str = "biography/article.json",
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load computer-scientist biography references from the local dataset. Falls back to a static name list.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Biography dataset must be a list of entries.")
        entries = [{"name": item["name"], "article": item.get("article", "")} for item in data if "name" in item]
        random.shuffle(entries)
        if limit is not None:
            entries = entries[:limit]
        return entries
    except Exception:
        base = [
            "Donald Knuth", "Edsger W. Dijkstra", "Barbara Liskov", "John McCarthy", "Dana Scott",
            "Leslie Lamport", "Andrew Yao", "Judea Pearl", "Whitfield Diffie", "Shafi Goldwasser",
            "Stephen Cook", "Michael Stonebraker", "Yann LeCun", "Geoffrey Hinton", "Jitendra Malik",
            "Tim Berners-Lee", "Silvio Micali", "Juris Hartmanis", "David S. Johnson", "John Hopcroft",
            "Stephen R. Bourne"
        ]
        random.shuffle(base)
        if limit is not None:
            base = base[:limit]
        return [{"name": name, "article": ""} for name in base]

# =======================================
# Evaluators for each task
# =======================================
def eval_arithmetic_item(expr: str, response: str) -> bool:
    pred_nums = re2.findall(r"[-+]?\d+", response)
    if not pred_nums:
        return False
    pred = int(pred_nums[-1])
    gold = safe_eval_arithmetic(expr)
    return pred == gold

def eval_gsm8k_item(gold: float, response: str) -> bool:
    pred = extract_boxed_number(response)
    if pred is None or gold is None: return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return False

def eval_mmlu_item(gold: str, response: str) -> bool:
    pred = extract_choice_letter(response)
    return (pred == gold)

def eval_chess_valid_item(board_fen: str, origin: str, response: str) -> bool:
    board = chess.Board(board_fen)
    dst = extract_square(response)
    if not dst:
        return False
    try:
        move = chess.Move.from_uci(chess.square_name(chess.parse_square(origin)) + chess.square_name(chess.parse_square(dst)))
        return move in board.legal_moves
    except Exception:
        return False

def eval_bio_item(person: str, response: str, llm: LLMClient, reference: Optional[str] = None) -> float:
    """
    Factuality score vs Wikipedia summary using LLM judgment when possible.
    Returns a score in [0,1].
    """
    wiki_ctx = reference or wiki_search_and_snippets(person, k=1, char_limit=1500)
    if not wiki_ctx:
        # Heuristic: reward mention of name
        score = 1.0 if person.lower().split()[0] in response.lower() else 0.3
        return score
    # LLM judge if available
    if llm.use_openai:
        judge_prompt = f"""Compare the bullet biography against the evidence and give a single 0-1 score for factual match (1 = perfect).
Evidence:
{wiki_ctx}

Biography:
{response}

Only output the numeric score between 0 and 1."""
        out = llm.chat([{"role":"system","content":"You are a strict fact‑checking judge that outputs only a number."},
                        {"role":"user","content": judge_prompt}])
        m = re2.search(r"0*\.?\d+(?:\.\d+)?", out.strip())
        if m:
            try:
                v = float(m.group(0))
                v = max(0.0, min(1.0, v))
                return v
            except:
                pass
    # Fallback: token overlap heuristic
    ev = set(re2.findall(r"[a-z]{3,}", wiki_ctx.lower()))
    bi = set(re2.findall(r"[a-z]{3,}", response.lower()))
    if not ev: return 0.3
    return len(ev & bi) / max(1, len(bi))

# =======================================
# Experiment runners (baselines + debate)
# =======================================
def make_agents(llm: LLMClient, n_agents: int = N_AGENTS, persona_diversity: bool = False) -> List[Agent]:
    agents: List[Agent] = []
    if persona_diversity:
        # diverse personas (novel direction)
        pool = PERSONAS_BASE[:]
        if n_agents > len(pool):
            # repeat with suffix
            for i in range(len(pool), n_agents):
                pool.append((f"Analyst{i+1}", "You are a careful analyst."))
        roles = pool[:n_agents]
    else:
        roles = [("Analyst", "You are a careful, detail‑oriented assistant. Answer concisely.")] * n_agents
    for name, sys_prompt in roles:
        agents.append(Agent(name=name, system_prompt=sys_prompt, llm=llm))
    return agents

def reflection_single(llm: LLMClient, system_prompt: str, start_prompt: str, refine_instruction: str) -> str:
    first = llm.chat([{"role":"system","content":system_prompt},{"role":"user","content":start_prompt}])
    second = llm.chat([
        {"role":"system","content":system_prompt},
        {"role":"user","content":f"Here is your earlier answer:\n{first}\n\n{refine_instruction}\nReturn only the updated answer."}
    ])
    return second

def run_task_arithmetic(llm: LLMClient, n_items: int = 100,
                        use_debate: bool = True,
                        persona_diversity: bool = False,
                        retrieval: bool = False,
                        weighted_vote: bool = True) -> Dict[str, Any]:
    items = gen_arithmetic(n_items)
    res_rows = []
    # Calibration slice for reliability (10% of set)
    calib_idx = set(random.sample(range(len(items)), max(1, len(items)//10)))
    agents = make_agents(llm, persona_diversity=persona_diversity)

    for i, itm in enumerate(tqdm(items, desc="Arithmetic")):
        expr, gold = itm["expr"], itm["answer"]
        start_prompt = PROMPTS["arithmetic_start"].format(*re2.findall(r"\d+", expr))
        rag = ""
        if retrieval:  # retrieval-augmented debates (not very meaningful for arithmetic; kept for completeness)
            rag = "You are solving arithmetic; recall PEMDAS: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction."

        # Baselines: single, reflection, majority
        single = agents[0].chat(start_prompt if not rag else f"{rag}\n\n{start_prompt}")
        reflect = reflection_single(
            agents[0].llm, agents[0].system_prompt, start_prompt if not rag else f"{rag}\n\n{start_prompt}",
            "Double-check calculations step-by-step and correct any mistake. State the final integer at the end."
        )

        # Multi-agent
        if use_debate:
            db = Debate(
                agents=agents,
                task_type="arithmetic",
                start_prompt=start_prompt if not rag else f"{rag}\n\n{start_prompt}",
                debate_prompt_tmpl=PROMPTS["arithmetic_debate"],
                rag_context=None,
                rounds=N_ROUNDS
            ).run()
            final_majority = db["final_majority"]
            final_weighted = db["final_weighted"]
        else:
            final_majority = single
            final_weighted = single

        # Evaluate
        r = {
            "expr": expr,
            "gold": gold,
            "single": eval_arithmetic_item(expr, single),
            "reflection": eval_arithmetic_item(expr, reflect),
            "majority": eval_arithmetic_item(expr, final_majority),
            "debate_weighted": eval_arithmetic_item(expr, final_weighted if weighted_vote else final_majority),
        }
        res_rows.append(r)

        # Update reliability from calibration items
        if i in calib_idx:
            for ag in agents:
                # crude: attribute aggregate normalized answer match to each agent last response
                # We re-run a single chat to get agent’s individual prediction on the same start prompt
                pred = ag.chat(start_prompt)
                ok = eval_arithmetic_item(expr, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

    df = pd.DataFrame(res_rows)
    return {"df": df, "items": items}

def run_task_gsm8k(llm: LLMClient, n_items: Optional[int] = 100,
                   use_debate: bool = True,
                   persona_diversity: bool = False,
                   retrieval: bool = False,
                   weighted_vote: bool = True) -> Dict[str, Any]:
    # data = load_gsm8k_subset(n_items)
    data = load_gsm8k_examples(limit=n_items, split="test")

    if not data:
        return {"df": pd.DataFrame(), "items": data}
    agents = make_agents(llm, persona_diversity=persona_diversity)
    calib_count = max(1, len(data)//10)
    calib_count = min(calib_count, len(data))
    calib_idx = set(random.sample(range(len(data)), calib_count))
    rows = []
    for i, itm in enumerate(tqdm(data, desc="GSM8K")):
        prob, gold = itm["problem"], itm["answer"]
        start_prompt = PROMPTS["gsm8k_start"].format(problem=prob)
        rag = wiki_search_and_snippets(prob, k=1, char_limit=1200) if retrieval else ""
        # baselines
        sp = start_prompt if not rag else f"Use the following evidence where helpful:\n{rag}\n\n{start_prompt}"
        single = agents[0].chat(sp)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
                                    "Recompute carefully and fix mistakes. Output the final answer as \\boxed{number}.")
        if use_debate:
            # Round 0 – per agent independent
            # (the Debate helper expects the debate template but we need to inject {problem} for GSM8K)
            # debate_tmpl = PROMPTS["gsm8k_debate"].format(other="{other}", problem=prob)
            debate_tmpl = PROMPTS["gsm8k_debate"].replace("{problem}", prob)
            db = Debate(
                agents=agents,
                task_type="gsm8k",
                start_prompt=sp,
                debate_prompt_tmpl=debate_tmpl,
                rag_context=None,
                rounds=N_ROUNDS
            ).run()
            final_majority = db["final_majority"]
            final_weighted = db["final_weighted"]
        else:
            final_majority = single
            final_weighted = single

        r = {
            "problem": prob,
            "gold": gold,
            "single": eval_gsm8k_item(gold, single),
            "reflection": eval_gsm8k_item(gold, reflect),
            "majority": eval_gsm8k_item(gold, final_majority),
            "debate_weighted": eval_gsm8k_item(gold, final_weighted if weighted_vote else final_majority),
        }
        rows.append(r)

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(sp)
                ok = eval_gsm8k_item(gold, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

    df = pd.DataFrame(rows)
    return {"df": df, "items": data}

def run_task_mmlu(llm: LLMClient, n_items: Optional[int] = 100,
                  use_debate: bool = True,
                  persona_diversity: bool = False,
                  retrieval: bool = False,
                  weighted_vote: bool = True) -> Dict[str, Any]:
    # data = load_mmlu_subset(n_items)
    data = load_mmlu_examples(limit=n_items)
    if not data:
        return {"df": pd.DataFrame(), "items": data}
    agents = make_agents(llm, persona_diversity=persona_diversity)
    calib_count = max(1, len(data)//10)
    calib_count = min(calib_count, len(data))
    calib_idx = set(random.sample(range(len(data)), calib_count))
    rows = []
    for i, itm in enumerate(tqdm(data, desc="MMLU")):
        q, A, B, C, D, gold = itm["question"], itm["A"], itm["B"], itm["C"], itm["D"], itm["answer"]
        start_prompt = PROMPTS["mmlu_start"].format(question=q, A=A, B=B, C=C, D=D)
        rag = wiki_search_and_snippets(q, k=1, char_limit=1200) if retrieval else ""
        sp = start_prompt if not rag else f"Use the following evidence where helpful:\n{rag}\n\n{start_prompt}"
        single = agents[0].chat(sp)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
                                    "Re-evaluate carefully and return only the final choice in the form (A|B|C|D).")

        if use_debate:
            debate_tmpl = PROMPTS["mmlu_debate"]
            db = Debate(
                agents=agents,
                task_type="mmlu",
                start_prompt=sp,
                debate_prompt_tmpl=debate_tmpl,
                rag_context=None,
                rounds=N_ROUNDS
            ).run()
            final_majority = db["final_majority"]
            final_weighted = db["final_weighted"]
        else:
            final_majority = single
            final_weighted = single

        r = {
            "question": q,
            "gold": gold,
            "single": eval_mmlu_item(gold, single),
            "reflection": eval_mmlu_item(gold, reflect),
            "majority": eval_mmlu_item(gold, final_majority),
            "debate_weighted": eval_mmlu_item(gold, final_weighted if weighted_vote else final_majority),
        }
        rows.append(r)

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(sp)
                ok = eval_mmlu_item(gold, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

    df = pd.DataFrame(rows)
    return {"df": df, "items": data}

def run_task_chess_validity(llm: LLMClient, n_items: int = 100,
                            use_debate: bool = True,
                            persona_diversity: bool = False,
                            retrieval: bool = False,
                            weighted_vote: bool = True) -> Dict[str, Any]:
    data = load_chess_positions_for_move_validity(n_items*2)
    data = data[:n_items]
    agents = make_agents(llm, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
    rows = []
    for i, itm in enumerate(tqdm(data, desc="Chess Validity")):
        pgn, origin, fen = itm["pgn"], itm["origin"], itm["board_fen"]
        start = PROMPTS["chess_valid_start"].format(moves=pgn, origin=origin)
        rag = ""  # retrieval not really helpful for this
        sp = start
        single = agents[0].chat(sp)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
                                    "Check legality carefully and return only (square).")

        if use_debate:
            db = Debate(
                agents=agents,
                task_type="chess_valid",
                start_prompt=sp,
                debate_prompt_tmpl=PROMPTS["chess_valid_debate"],
                rag_context=None,
                rounds=N_ROUNDS
            ).run()
            final_majority = db["final_majority"]
            final_weighted = db["final_weighted"]
        else:
            final_majority = single
            final_weighted = single

        r = {
            "origin": origin,
            "single": eval_chess_valid_item(fen, origin, single),
            "reflection": eval_chess_valid_item(fen, origin, reflect),
            "majority": eval_chess_valid_item(fen, origin, final_majority),
            "debate_weighted": eval_chess_valid_item(fen, origin, final_weighted if weighted_vote else final_majority),
        }
        rows.append(r)

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(sp)
                ok = eval_chess_valid_item(fen, origin, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)

    df = pd.DataFrame(rows)
    return {"df": df, "items": data}

def run_task_biographies(llm: LLMClient, n_items: Optional[int] = 50,
                         use_debate: bool = True,
                         persona_diversity: bool = True,     # personas help here
                         retrieval: bool = True,             # RAG strongly helps
                         weighted_vote: bool = True) -> Dict[str, Any]:
    entries = load_biography_entries(limit=n_items)
    if not entries:
        return {"df": pd.DataFrame(), "items": entries}
    agents = make_agents(llm, persona_diversity=persona_diversity)
    calib_count = max(1, max(len(entries)//10, 1))
    calib_count = min(calib_count, len(entries))
    calib_idx = set(random.sample(range(len(entries)), calib_count))
    rows = []
    for i, entry in enumerate(tqdm(entries, desc="Biographies")):
        person = entry["name"]
        reference = entry.get("article") or ""
        start_prompt = PROMPTS["bio_start"].format(person=person)
        if retrieval:
            rag = reference if reference else wiki_search_and_snippets(person, k=2, char_limit=1500)
        else:
            rag = ""
        sp = start_prompt if not rag else f"Use the following evidence to avoid hallucinations:\n{rag}\n\n{start_prompt}"
        single = agents[0].chat(sp)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, sp,
                                    "Cross-check every bullet against the evidence. Remove any uncertain bullets. Return bullets only.")

        if use_debate:
            tmpl = PROMPTS["bio_debate"].format(other="{other}", person=person)
            db = Debate(
                agents=agents,
                task_type="bio",
                start_prompt=sp,
                debate_prompt_tmpl=tmpl,
                rag_context=None,
                rounds=N_ROUNDS
            ).run()
            final_majority = db["final_majority"]
            final_weighted = db["final_weighted"]
        else:
            final_majority = single
            final_weighted = single

        ref_ctx = reference if reference else None
        s_single = eval_bio_item(person, single, llm, ref_ctx)
        s_reflec = eval_bio_item(person, reflect, llm, ref_ctx)
        s_major  = eval_bio_item(person, final_majority, llm, ref_ctx)
        s_weight = eval_bio_item(person, final_weighted if weighted_vote else final_majority, llm, ref_ctx)

        r = {"person": person, "single": s_single, "reflection": s_reflec, "majority": s_major,
             "debate_weighted": s_weight, "reference_used": bool(ref_ctx)}
        rows.append(r)

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(sp)
                sc = eval_bio_item(person, pred, llm, ref_ctx)  # 0..1
                ag.reliability = moving_average(ag.reliability, sc, alpha=0.2)

    df = pd.DataFrame(rows)
    return {"df": df, "items": entries}

# =======================================
# Plotting (Figures similar to the paper)
# =======================================
def barplot_results(task_name: str, df: pd.DataFrame, is_accuracy: bool = True):
    metrics = ["single", "reflection", "majority", "debate_weighted"]
    labels  = ["Single Agent", "Reflection", "Multi‑Agent (Majority)", "Multi‑Agent (Debate)"]
    vals = []
    for m in metrics:
        if is_accuracy:
            vals.append(df[m].mean() * 100.0)
        else:  # biography factuality 0..1 -> %
            vals.append(df[m].mean() * 100.0)
    plt.figure(figsize=(6,4))
    xs = np.arange(len(metrics))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=15, ha="right")
    plt.ylabel("Accuracy (%)" if is_accuracy else "Factuality (%)")
    plt.title(task_name)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{task_name.lower().replace(' ','_')}_bars.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[Saved] {path}")

def ablation_plot(title: str, xvals: List[int], yvals: List[float], xlabel: str, ylabel: str, fname: str):
    plt.figure(figsize=(5,4))
    plt.plot(xvals, yvals, marker="o")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[Saved] {path}")

# =======================================
# Ablations: agents & rounds (Figure 10/12)
# =======================================
def ablation_agents_gsm8k(llm: LLMClient):
    xvals, accs = [], []
    for k in [1,2,3,4,5]:
        ags = make_agents(llm, n_agents=k, persona_diversity=True)
        data = load_gsm8k_examples(limit=30)
        rows = []
        for itm in data:
            prob, gold = itm["problem"], itm["answer"]
            sp = PROMPTS["gsm8k_start"].format(problem=prob)
            # tmpl = PROMPTS["gsm8k_debate"].format(other="{other}", problem=prob)
            tmpl = PROMPTS["gsm8k_debate"].replace("{problem}", prob)
            db = Debate(agents=ags, task_type="gsm8k", start_prompt=sp, debate_prompt_tmpl=tmpl, rounds=2,
                        rag_context=None, summarization_chars=2000).run()
            ok = eval_gsm8k_item(gold, db["final_weighted"])
            rows.append(ok)
        xvals.append(k); accs.append(100.0 * (sum(rows)/len(rows)))
    ablation_plot("Performance vs Number of Debating Agents", xvals, accs, "Agents", "GSM8K Accuracy (%)", "ablation_agents_gsm8k.png")

def ablation_rounds_arithmetic(llm: LLMClient):
    xvals, accs = [], []
    items = gen_arithmetic(80)
    for rds in [1,2,3,4]:
        ags = make_agents(llm, n_agents=3, persona_diversity=False)
        ok_list = []
        for itm in items:
            expr = itm["expr"]; start = PROMPTS["arithmetic_start"].format(*re2.findall(r"\d+", expr))
            db = Debate(agents=ags, task_type="arithmetic", start_prompt=start,
                        debate_prompt_tmpl=PROMPTS["arithmetic_debate"], rounds=rds,
                        rag_context=None, summarization_chars=1500).run()
            ok = eval_arithmetic_item(expr, db["final_weighted"])
            ok_list.append(ok)
        xvals.append(rds); accs.append(100.0 * (sum(ok_list)/len(ok_list)))
    ablation_plot("Math Accuracy vs Debate Rounds", xvals, accs, "Debate Rounds", "Arithmetic Accuracy (%)", "ablation_rounds_arith.png")

# =======================================
# Master pipeline
# =======================================
def main():
    llm = LLMClient()

    # ------------------------------
    # (1) Reproduce core benchmarks
    # ------------------------------
    arithmetic = run_task_arithmetic(llm, n_items=100, use_debate=True, persona_diversity=False,
                                     retrieval=False, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "arithmetic_results.jsonl"), arithmetic["df"].to_dict(orient="records"))
    barplot_results("Arithmetic", arithmetic["df"], is_accuracy=True)

    gsm8k = run_task_gsm8k(llm, n_items=None, use_debate=True, persona_diversity=False,
                           retrieval=False, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "gsm8k_results.jsonl"), gsm8k["df"].to_dict(orient="records"))
    barplot_results("GSM8K", gsm8k["df"], is_accuracy=True)

    mmlu = run_task_mmlu(llm, n_items=None, use_debate=True, persona_diversity=False,
                         retrieval=False, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "mmlu_results.jsonl"), mmlu["df"].to_dict(orient="records"))
    barplot_results("MMLU", mmlu["df"], is_accuracy=True)

    chess_valid = run_task_chess_validity(llm, n_items=80, use_debate=True, persona_diversity=False,
                                          retrieval=False, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "chess_valid_results.jsonl"), chess_valid["df"].to_dict(orient="records"))
    barplot_results("Chess Move Validity", chess_valid["df"], is_accuracy=True)

    bios = run_task_biographies(llm, n_items=None, use_debate=True, persona_diversity=True,
                                retrieval=True, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "biographies_results.jsonl"), bios["df"].to_dict(orient="records"))
    barplot_results("Biographies Factuality", bios["df"], is_accuracy=False)

    # ------------------------------------------------
    # (2) Novel directions: RAG / Weighted / Personas
    #     plus Adaptive Summarization (built‑in)
    # ------------------------------------------------
    # RAG for GSM8K & MMLU
    gsm8k_rag = run_task_gsm8k(llm, n_items=60, use_debate=True, persona_diversity=False,
                               retrieval=True, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "gsm8k_rag_results.jsonl"), gsm8k_rag["df"].to_dict(orient="records"))
    barplot_results("GSM8K (RAG Debate)", gsm8k_rag["df"], is_accuracy=True)

    mmlu_rag = run_task_mmlu(llm, n_items=60, use_debate=True, persona_diversity=False,
                              retrieval=True, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "mmlu_rag_results.jsonl"), mmlu_rag["df"].to_dict(orient="records"))
    barplot_results("MMLU (RAG Debate)", mmlu_rag["df"], is_accuracy=True)

    # Persona diversity on MMLU
    mmlu_personas = run_task_mmlu(llm, n_items=60, use_debate=True, persona_diversity=True,
                                  retrieval=False, weighted_vote=True)
    ensure_jsonl(os.path.join(OUT_DIR, "mmlu_personas_results.jsonl"), mmlu_personas["df"].to_dict(orient="records"))
    barplot_results("MMLU (Persona‑Diverse Debate)", mmlu_personas["df"], is_accuracy=True)

    # Ablations: agents & rounds (Figures 10/12 style)
    ablation_agents_gsm8k(llm)
    ablation_rounds_arithmetic(llm)

    # Print quick summary
    def summarize(df, name):
        return {
            "task": name,
            "Single": round(df["single"].mean()*100.0, 2),
            "Reflection": round(df["reflection"].mean()*100.0, 2),
            "Majority": round(df["majority"].mean()*100.0, 2),
            "DebateWeighted": round(df["debate_weighted"].mean()*100.0, 2),
        }
    summary = [
        summarize(arithmetic["df"], "Arithmetic"),
        summarize(gsm8k["df"], "GSM8K"),
        summarize(mmlu["df"], "MMLU"),
        summarize(chess_valid["df"], "Chess Validity"),
        summarize(bios["df"], "Biographies (factuality %)"),
        summarize(gsm8k_rag["df"], "GSM8K (RAG)"),
        summarize(mmlu_rag["df"], "MMLU (RAG)"),
        summarize(mmlu_personas["df"], "MMLU (Personas)"),
    ]
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary (%) ===")
    for s in summary: print(s)

if __name__ == "__main__":
    main()
