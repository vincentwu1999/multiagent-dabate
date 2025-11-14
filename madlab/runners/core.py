# madlab/runners/core.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import random
import pandas as pd
import json as _json
import regex as re
import chess

from ..llm import LLMClient
from ..agents import make_agents
from ..debate import Debate
from ..prompts import PROMPTS
from ..utils import moving_average
from ..evals import (
    eval_arithmetic_item, eval_gsm8k_item, eval_mmlu_item, eval_chess_valid_item, eval_bio_item
)
from ..data.arithmetic import gen_arithmetic
from ..data.gsm8k import load_gsm8k_examples
from ..data.mmlu import load_mmlu_examples
from ..data.chess_valid import load_chess_positions_for_move_validity
from ..data.bio import load_biography_entries
from ..data.bioasq import load_bioasq_examples
from ..evals import eval_bioasq_item 

from ..aggregation.weights import REGISTRY as WEIGHT_REGISTRY, ConfidenceCalibrator, parse_confidence_pct

# ---------------- Helpers ----------------
def _pack_json(obj) -> str:
    try:
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""

def _agent_meta(agents) -> List[Dict[str, Any]]:
    return [{"name": a.name, "reliability": float(getattr(a, "reliability", 0.0))} for a in agents]

def _mk_conf_calibrator() -> ConfidenceCalibrator:
    return ConfidenceCalibrator(bins=10, alpha=0.05)

def _task_verifier(task_type: str, gold_ctx: Dict[str, Any]):
    if task_type == "arithmetic":
        expr = gold_ctx["expr"]
        return lambda r: bool(eval_arithmetic_item(expr, r))
    if task_type == "gsm8k":
        gold = gold_ctx["gold"]
        return lambda r: bool(eval_gsm8k_item(gold, r))
    if task_type == "mmlu":
        gold = gold_ctx["gold"]
        return lambda r: bool(eval_mmlu_item(gold, r))
    if task_type == "chess_valid":
        fen, origin = gold_ctx["fen"], gold_ctx["origin"]
        return lambda r: bool(eval_chess_valid_item(fen, origin, r))
    # bios is continuous factuality; treat ≥0.7 as “ok” if desired
    if task_type == "bio":
        person = gold_ctx["person"]
        return lambda r: bool(eval_bio_item(person, r) >= 0.7)
    return lambda r: False

# -------- optional chess engine for "best move" --------
import os
import chess.engine

def _maybe_open_engine() -> Optional[chess.engine.SimpleEngine]:
    path = os.environ.get("STOCKFISH_PATH", "")
    if not path:
        return None
    try:
        return chess.engine.SimpleEngine.popen_uci(path)
    except Exception:
        return None

def _score_move_with_engine(fen: str, san_or_uci: str, engine: chess.engine.SimpleEngine, movetime_ms: int = 200):
    try:
        board = chess.Board(fen)
        text = san_or_uci.strip()
        m = re.search(r"\b\d+\.\s*(.+)", text)
        if m:
            text = m.group(1).strip()
        try:
            move = board.parse_san(text)
        except Exception:
            move = chess.Move.from_uci(text.replace(" ", ""))
            if move not in board.legal_moves:
                return False, None
        if move not in board.legal_moves:
            return False, None

        limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        info_before = engine.analyse(board, limit)
        score_before = info_before.get("score")
        board.push(move)
        info_after = engine.analyse(board, limit)
        score_after = info_after.get("score")

        def _cp(s):
            try:
                return s.white().score(mate_score=100000)
            except Exception:
                return None
        cb, ca = _cp(score_before), _cp(score_after)
        if cb is None or ca is None:
            return True, None
        return True, float(ca - cb)
    except Exception:
        return False, None

# --------------- Reflection helper ---------------
def reflection_single(llm: LLMClient, system_prompt: str, start_prompt: str, refine_instruction: str) -> str:
    first = llm.chat([{"role": "system", "content": system_prompt},
                      {"role": "user", "content": start_prompt}])
    second = llm.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user",  "content": f"Here is your earlier answer:\n{first}\n\n{refine_instruction}\nReturn only the updated answer."}
    ])
    return second

# ===================== Arithmetic =====================
def run_task_arithmetic(
    llm: LLMClient,
    n_items: int = 50,
    use_debate: bool = True,
    persona_diversity: bool = False,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
) -> Dict[str, Any]:
    items = gen_arithmetic(n_items)
    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(items)), max(1, len(items)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])

    rows = []
    for i, itm in enumerate(items):
        expr, gold = itm["expr"], itm["answer"]
        start_prompt = PROMPTS["arithmetic_start"].format(*re.findall(r"\d+", expr))

        single = agents[0].chat(start_prompt)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, start_prompt,
                                    "Double-check calculations step-by-step and correct any mistake. State the final integer at the end.")

        db = None; agent_texts=[]; history=[]
        if use_debate:
            weight_ctx = {
                "verifier": _task_verifier("arithmetic", {"expr": expr}),
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            db = Debate(
                agents=agents, task_type="arithmetic",
                start_prompt=start_prompt, debate_prompt_tmpl=PROMPTS["arithmetic_debate"],
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        rows.append({
            "expr": expr, "gold": gold,
            "single": eval_arithmetic_item(expr, single),
            "reflection": eval_arithmetic_item(expr, reflect),
            "majority": eval_arithmetic_item(expr, final_majority),
            "debate_weighted": eval_arithmetic_item(expr, final_weighted),
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start_prompt)
                ok = eval_arithmetic_item(expr, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), ok)

    return {"df": pd.DataFrame(rows), "items": items}

# ===================== GSM8K =====================
def run_task_gsm8k(
    llm: LLMClient,
    n_items: Optional[int] = 50,
    use_debate: bool = True,
    persona_diversity: bool = False,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
) -> Dict[str, Any]:
    data = load_gsm8k_examples(limit=n_items)
    if not data: return {"df": pd.DataFrame(), "items": data}

    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])

    rows = []
    for i, itm in enumerate(data):
        prob, gold = itm["problem"], itm["answer"]
        start_prompt = PROMPTS["gsm8k_start"].format(problem=prob)

        single = agents[0].chat(start_prompt)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, start_prompt,
                                    "Recompute carefully and fix mistakes. Output the final answer as \\boxed{number}.")

        db=None; agent_texts=[]; history=[]
        if use_debate:
            weight_ctx = {
                "verifier": _task_verifier("gsm8k", {"gold": gold}),
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            tmpl = PROMPTS["gsm8k_debate"].replace("{problem}", prob)
            db = Debate(
                agents=agents, task_type="gsm8k",
                start_prompt=start_prompt, debate_prompt_tmpl=tmpl,
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        rows.append({
            "problem": prob, "gold": gold,
            "single": eval_gsm8k_item(gold, single),
            "reflection": eval_gsm8k_item(gold, reflect),
            "majority": eval_gsm8k_item(gold, final_majority),
            "debate_weighted": eval_gsm8k_item(gold, final_weighted),
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start_prompt)
                ok = eval_gsm8k_item(gold, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), ok)

    return {"df": pd.DataFrame(rows), "items": data}

# ===================== MMLU =====================
def run_task_mmlu(
    llm: LLMClient,
    n_items: Optional[int] = 50,
    use_debate: bool = True,
    persona_diversity: bool = False,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
) -> Dict[str, Any]:
    data = load_mmlu_examples(limit=n_items)
    if not data: return {"df": pd.DataFrame(), "items": data}

    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])

    rows = []
    for i, itm in enumerate(data):
        q, A, B, C, D, gold = itm["question"], itm["A"], itm["B"], itm["C"], itm["D"], itm["answer"]
        start_prompt = PROMPTS["mmlu_start"].format(question=q, A=A, B=B, C=C, D=D)

        single = agents[0].chat(start_prompt)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, start_prompt,
                                    "Re-evaluate carefully and return only the final choice in the form (A|B|C|D).")

        db=None; agent_texts=[]; history=[]
        if use_debate:
            weight_ctx = {
                "verifier": _task_verifier("mmlu", {"gold": gold}),
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            db = Debate(
                agents=agents, task_type="mmlu",
                start_prompt=start_prompt, debate_prompt_tmpl=PROMPTS["mmlu_debate"],
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        rows.append({
            "question": q, "gold": gold,
            "single": eval_mmlu_item(gold, single),
            "reflection": eval_mmlu_item(gold, reflect),
            "majority": eval_mmlu_item(gold, final_majority),
            "debate_weighted": eval_mmlu_item(gold, final_weighted),
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
            "A": A, "B": B, "C": C, "D": D, "subject": itm.get("subject", "")
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start_prompt)
                ok = eval_mmlu_item(gold, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), ok)

    return {"df": pd.DataFrame(rows), "items": data}

# ===================== Chess: Valid Destination =====================
def run_task_chess_validity(
    llm: LLMClient,
    n_items: int = 50,
    use_debate: bool = True,
    persona_diversity: bool = False,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
) -> Dict[str, Any]:
    data = load_chess_positions_for_move_validity(n_items*2)[:n_items]
    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])

    rows = []
    for i, itm in enumerate(data):
        pgn, origin, fen = itm["pgn"], itm["origin"], itm["board_fen"]
        start = PROMPTS["chess_valid_start"].format(moves=pgn, origin=origin)

        single = agents[0].chat(start)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, start, "Check legality carefully and return only (square).")

        db=None; agent_texts=[]; history=[]
        if use_debate:
            weight_ctx = {
                "verifier": _task_verifier("chess_valid", {"fen": fen, "origin": origin}),
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            db = Debate(
                agents=agents, task_type="chess_valid",
                start_prompt=start, debate_prompt_tmpl=PROMPTS["chess_valid_debate"],
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        rows.append({
            "origin": origin, "fen": fen, "pgn": pgn,
            "single": eval_chess_valid_item(fen, origin, single),
            "reflection": eval_chess_valid_item(fen, origin, reflect),
            "majority": eval_chess_valid_item(fen, origin, final_majority),
            "debate_weighted": eval_chess_valid_item(fen, origin, final_weighted),
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start)
                ok = eval_chess_valid_item(fen, origin, pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), ok)

    return {"df": pd.DataFrame(rows), "items": data}

# ===================== Chess: Best Move =====================
def run_task_chess_move(
    llm: LLMClient,
    n_items: int = 50,
    use_debate: bool = True,
    persona_diversity: bool = False,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
    use_engine: bool = True,
) -> Dict[str, Any]:
    raw = load_chess_positions_for_move_validity(n_items*2)
    data = raw[:n_items]
    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])
    engine = _maybe_open_engine() if use_engine else None

    rows = []
    for i, itm in enumerate(data):
        pgn, fen = itm["pgn"], itm["board_fen"]
        start_prompt = PROMPTS["chess_move_start"].format(moves=pgn)

        single = agents[0].chat(start_prompt)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, start_prompt,
                                    "Reconsider tactical/positional ideas and return only a single move of the form '14. <MOVE>'.")

        db=None; agent_texts=[]; history=[]
        if use_debate:
            weight_ctx = {
                # no strict verifier for optimality; optionally treat legality as verifier
                "verifier": None,
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            db = Debate(
                agents=agents, task_type="chess_move",
                start_prompt=start_prompt, debate_prompt_tmpl=PROMPTS["chess_move_debate"],
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        def _legal_and_score(text: str):
            if engine is None:
                try:
                    board = chess.Board(fen)
                    t = text.strip()
                    m = re.search(r"\b\d+\.\s*(.+)", t)
                    if m: t = m.group(1).strip()
                    try:
                        mv = board.parse_san(t)
                    except Exception:
                        mv = chess.Move.from_uci(t.replace(" ", ""))
                    return (mv in board.legal_moves, None)
                except Exception:
                    return (False, None)
            legal, cp = _score_move_with_engine(fen, text, engine=engine)
            return (bool(legal), cp)

        s_leg, s_cp = _legal_and_score(single)
        r_leg, r_cp = _legal_and_score(reflect)
        m_leg, m_cp = _legal_and_score(final_majority)
        w_leg, w_cp = _legal_and_score(final_weighted)

        rows.append({
            "fen": fen, "pgn": pgn,
            "single": bool(s_leg), "reflection": bool(r_leg),
            "majority": bool(m_leg), "debate_weighted": bool(w_leg),
            "single_cp_improvement": s_cp, "reflection_cp_improvement": r_cp,
            "majority_cp_improvement": m_cp, "debate_weighted_cp_improvement": w_cp,
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start_prompt)
                # calibration on legality only
                leg, _ = _legal_and_score(pred)
                ag.reliability = moving_average(ag.reliability, 1.0 if leg else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), bool(leg))

    if engine is not None:
        try: engine.quit()
        except Exception: pass

    return {"df": pd.DataFrame(rows), "items": data}

# ===================== Biographies =====================
def run_task_biographies(
    llm: LLMClient,
    n_items: Optional[int] = 50,
    use_debate: bool = True,
    persona_diversity: bool = True,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
) -> Dict[str, Any]:
    entries = load_biography_entries(limit=n_items)
    if not entries: return {"df": pd.DataFrame(), "items": entries}

    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(entries)), max(1, len(entries)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])

    rows = []
    for i, entry in enumerate(entries):
        person = entry["name"]
        start_prompt = PROMPTS["bio_start"].format(person=person)

        single = agents[0].chat(start_prompt)
        reflect = reflection_single(agents[0].llm, agents[0].system_prompt, start_prompt,
                                    "Remove uncertain bullets and keep only factual items. Return bullets only.")

        db=None; agent_texts=[]; history=[]
        if use_debate:
            weight_ctx = {
                "verifier": _task_verifier("bio", {"person": person}),
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            tmpl = PROMPTS["bio_debate"].format(other="{other}", person=person)
            db = Debate(
                agents=agents, task_type="bio",
                start_prompt=start_prompt, debate_prompt_tmpl=tmpl,
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        s_single = eval_bio_item(person, single)
        s_reflec = eval_bio_item(person, reflect)
        s_major  = eval_bio_item(person, final_majority)
        s_weight = eval_bio_item(person, final_weighted)

        rows.append({
            "person": person, "reference_used": False,
            "single": s_single, "reflection": s_reflec,
            "majority": s_major, "debate_weighted": s_weight,
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start_prompt)
                ok = bool(eval_bio_item(person, pred) >= 0.7)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), ok)

    return {"df": pd.DataFrame(rows), "items": entries}

# BIoasq
def run_task_bioasq(
    llm: LLMClient,
    n_items: Optional[int] = 50,
    subset: str = "all",                  # "factoid" | "yesno" | "list" | "all"
    use_debate: bool = True,
    persona_diversity: bool = True,
    use_weighted: bool = True,
    weight_rule: str = "ensemble",
    conf_cal: ConfidenceCalibrator | None = None,
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2, delta: float = 0.15,
) -> Dict[str, Any]:
    data = load_bioasq_examples(limit=n_items, subset=subset)
    if not data: 
        return {"df": pd.DataFrame(), "items": data}

    agents = make_agents(llm, n_agents=3, persona_diversity=persona_diversity)
    calib_idx = set(random.sample(range(len(data)), max(1, len(data)//10)))
    conf_cal = conf_cal or _mk_conf_calibrator()
    weight_fn = WEIGHT_REGISTRY.get(weight_rule, WEIGHT_REGISTRY["rolling_acc"])

    rows = []
    for i, ex in enumerate(data):
        qtype, question, gold = ex["type"], ex["question"], ex["answers"]
        start_prompt = PROMPTS["bioasq_start"].format(question=question, qtype=qtype)

        single = agents[0].chat(start_prompt)
        reflect = reflection_single(
            agents[0].llm, agents[0].system_prompt, start_prompt,
            "Re-check biomedical facts and return only the FINAL line in 'FINAL: <answer>' format."
        )

        db=None; agent_texts=[]; history=[]
        if use_debate:
            # verifier: treat score >=0.9 as pass (strict for factoid/yesno; list is F1)
            def _vf(resp: str) -> bool:
                sc = eval_bioasq_item(qtype, gold, resp)
                return sc >= (0.9 if qtype in {"factoid","yesno"} else 0.7)

            weight_ctx = {
                "verifier": _vf,
                "conf_calibrator": conf_cal,
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            }
            db = Debate(
                agents=agents, task_type="bioasq",
                start_prompt=start_prompt, debate_prompt_tmpl=PROMPTS["bioasq_debate"],
                rounds=2, weight_fn=weight_fn, weight_ctx=weight_ctx, use_weighted=use_weighted
            ).run()
            final_majority = db["final_majority"]; final_weighted = db["final_weighted"]
            agent_texts = db.get("all_final", []); history = db.get("history", [])
        else:
            final_majority = final_weighted = single

        s_single = eval_bioasq_item(qtype, gold, single)
        s_reflec = eval_bioasq_item(qtype, gold, reflect)
        s_major  = eval_bioasq_item(qtype, gold, final_majority)
        s_weight = eval_bioasq_item(qtype, gold, final_weighted)

        rows.append({
            "type": qtype, "question": question, "gold_answers": _pack_json(gold),
            "single": s_single, "reflection": s_reflec, "majority": s_major, "debate_weighted": s_weight,
            "single_text": single, "reflection_text": reflect,
            "majority_text": final_majority, "debate_weighted_text": final_weighted,
            "agent_meta": _pack_json(_agent_meta(agents)),
            "agent_final_texts": _pack_json(agent_texts),
            "debate_history": _pack_json(history),
        })

        if i in calib_idx:
            for ag in agents:
                pred = ag.chat(start_prompt)
                ok = eval_bioasq_item(qtype, gold, pred) >= (0.9 if qtype in {"factoid","yesno"} else 0.7)
                ag.reliability = moving_average(ag.reliability, 1.0 if ok else 0.0, alpha=0.2)
                conf_cal.update(parse_confidence_pct(pred), bool(ok))

    return {"df": pd.DataFrame(rows), "items": data}
