from typing import Optional
import chess
from .utils import extract_boxed_number, extract_choice_letter, safe_eval_arithmetic, extract_square

def eval_arithmetic_item(expr: str, response: str) -> bool:
    import regex as re
    nums = re.findall(r"[-+]?\d+", response)
    if not nums: return False
    pred = int(nums[-1])
    gold = safe_eval_arithmetic(expr)
    return pred == gold

def eval_gsm8k_item(gold: float, response: str) -> bool:
    pred = extract_boxed_number(response)
    if pred is None or gold is None: return False
    try: return abs(float(pred) - float(gold)) < 1e-6
    except Exception: return False

def eval_mmlu_item(gold: str, response: str) -> bool:
    pred = extract_choice_letter(response)
    return (pred == gold)

def eval_chess_valid_item(board_fen: str, origin: str, response: str) -> bool:
    board = chess.Board(board_fen)
    dst = extract_square(response)
    if not dst: return False
    try:
        move = chess.Move.from_uci(origin + dst)
        return move in board.legal_moves
    except Exception:
        return False

def eval_bio_item(person: str, response: str) -> float:
    # Lightweight lexical overlap proxy (works without API key)
    import regex as re
    bi = set(re.findall(r"[a-z]{3,}", response.lower()))
    return min(1.0, max(0.0, len(bi) / max(1, len(response.split()))))

# --- BioASQ style scoring ---
import regex as re

def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\.\-/%]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_final_line(text: str) -> str:
    # Try to read a final answer – looks for lines like "FINAL: <answer>"
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"^\s*final\s*:\s*(.+)$", ln, flags=re.I)
        if m: 
            return m.group(1).strip()
    # fallback: last non-empty line
    return lines[-1] if lines else ""

def _token_set(s: str) -> set:
    toks = re.findall(r"[a-z0-9]+", _normalize_text(s))
    return set(toks)

def eval_bioasq_item(qtype: str, gold_answers: list[str], response: str) -> float:
    """
    Returns a number in [0,1].
    - factoid: Exact/soft match against any gold after normalization (1 or 0)
    - yesno: yes/no match (1 or 0)
    - list: F1 against set of gold items
    The caller may aggregate across items -> accuracy or mean F1.
    """
    qtype = (qtype or "").lower()
    pred = _extract_final_line(response)

    # YES/NO
    if qtype == "yesno":
        g = "yes" if (gold_answers and str(gold_answers[0]).lower().startswith("y")) else "no"
        p = "yes" if _normalize_text(pred).startswith("yes") else ("no" if _normalize_text(pred).startswith("no") else _normalize_text(pred))
        return 1.0 if p == g else 0.0

    # FACTOID (soft exact)
    if qtype == "factoid":
        pnorm = _normalize_text(pred)
        for g in gold_answers:
            if pnorm == _normalize_text(g):
                return 1.0
        # allow small soft match: token overlap ≥ 0.8 of shorter
        ptok = _token_set(pred)
        for g in gold_answers:
            gtok = _token_set(g)
            if not ptok or not gtok: 
                continue
            overlap = len(ptok & gtok) / max(1, min(len(ptok), len(gtok)))
            if overlap >= 0.8:
                return 1.0
        return 0.0

    # LIST (F1 on unique, normalized tokens per item)
    if qtype == "list":
        gold_norm = { _normalize_text(g) for g in gold_answers if g.strip() }
        # allow comma/semicolon separated pred
        cand = [x.strip() for x in re.split(r"[,;]| and ", pred) if x.strip()]
        pred_norm = { _normalize_text(x) for x in cand if x.strip() }
        if not gold_norm and not pred_norm: 
            return 1.0
        if not gold_norm or not pred_norm: 
            return 0.0
        tp = len(gold_norm & pred_norm)
        prec = tp / max(1, len(pred_norm))
        rec  = tp / max(1, len(gold_norm))
        if prec+rec == 0: 
            return 0.0
        return 2*prec*rec/(prec+rec)

    # Unknown type: return 0
    return 0.0
