import json, random, numpy as np
import regex as re
from typing import Optional

random.seed(7); np.random.seed(7)

_DEF_TRUNC = 4000

def ensure_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def truncate(s: str, max_chars: int = _DEF_TRUNC) -> str:
    if len(s) <= max_chars: return s
    return s[:max_chars//2] + "\n[...snip...]\n" + s[-max_chars//2:]

def moving_average(prev: float, new: float, alpha: float = 0.2) -> float:
    return (1 - alpha) * prev + alpha * new

# Small parsers/extractors
num_re = re.compile(r"[-+]?\d*\.?\d+")
int_re = re.compile(r"[-+]?\d+")
choice_re = re.compile(r"\(([A-D])\)\s*$")
boxed_re = re.compile(r"\\boxed\{([^}]+)\}")
square_re = re.compile(r"\b([a-h][1-8])\b", re.I)

def safe_eval_arithmetic(expr: str) -> Optional[int]:
    if not re.fullmatch(r"[\d+\-*\s]+", expr):
        return None
    try:
        return int(eval(expr))
    except Exception:
        return None

def extract_boxed_number(text: str):
    text = text.strip()
    m = boxed_re.search(text)
    cand = m.group(1) if m else (num_re.findall(text)[-1] if num_re.findall(text) else None)
    if cand is None: return None
    try:
        return int(cand) if "." not in cand else float(cand)
    except Exception:
        try: return float(cand)
        except: return None

def extract_choice_letter(text: str):
    m = choice_re.search(text.strip())
    return m.group(1) if m else None

def extract_square(text: str):
    m = square_re.findall(text.strip())
    return m[-1].lower() if m else None
