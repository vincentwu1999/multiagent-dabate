# madlab/aggregation/weights.py
from __future__ import annotations
from typing import List, Dict, Any
import math, regex as re
import numpy as np

# ---- Confidence parsing & calibration ---------------------------------
def parse_confidence_pct(text: str) -> float:
    """
    Returns [0,1]. Looks for 'Confidence: XX%' or 'XX% confident' or 'confidence: 0.7'.
    Defaults to 0.5 if nothing found.
    """
    t = text.lower()
    m = re.search(r'confidence[:\s]*([0-1](?:\.\d+)?)', t)
    if m:
        v = float(m.group(1))
        return float(max(0.01, min(0.99, v)))
    m = re.search(r'(\d{1,3})\s*%(\s*confident)?', t)
    if m:
        v = float(m.group(1)) / 100.0
        return float(max(0.01, min(0.99, v)))
    return 0.5

class ConfidenceCalibrator:
    """
    Tiny monotonic calibrator: keeps running (p, ok) buckets and returns a smoothed mapping.
    Call update(p, ok) on calibration items; call apply(p) to get calibrated p'.
    """
    def __init__(self, bins: int = 10, alpha: float = 0.05):
        self.bins = bins
        self.alpha = alpha
        self.pos = np.zeros(bins, dtype=float)
        self.cnt = np.zeros(bins, dtype=float)

    def _bin(self, p: float) -> int:
        i = int(min(self.bins - 1, max(0, math.floor(p * self.bins))))
        return i

    def update(self, p: float, ok: bool):
        i = self._bin(p)
        self.cnt[i] += 1.0
        self.pos[i] += 1.0 if ok else 0.0

    def apply(self, p: float) -> float:
        if self.cnt.sum() < 5:
            return p
        i = self._bin(p)
        # Laplace smoothing + cumulative monotone pass
        rate = (self.pos + self.alpha) / (self.cnt + 2 * self.alpha)
        # enforce monotonic non-decreasing
        mono = np.maximum.accumulate(rate)
        return float(mono[i])

# ---- Similarity penalty (bag-of-words cosine) -------------------------
def _bow(text: str) -> Dict[str, int]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    bag: Dict[str, int] = {}
    for w in toks:
        bag[w] = bag.get(w, 0) + 1
    return bag

def _cos(a: Dict[str,int], b: Dict[str,int]) -> float:
    if not a or not b: return 0.0
    keys = set(a) | set(b)
    va = np.array([a.get(k,0) for k in keys], dtype=float)
    vb = np.array([b.get(k,0) for k in keys], dtype=float)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na == 0 or nb == 0: return 0.0
    return float(va.dot(vb) / (na*nb))

def similarity_penalty(responses: List[str]) -> List[float]:
    """Return per-agent similarity to centroid (higher = more similar)."""
    bows = [_bow(r) for r in responses]
    # centroid = average vector
    keys = set()
    for b in bows: keys |= set(b)
    if not keys:
        return [0.0]*len(responses)
    mats = np.array([[b.get(k,0) for k in keys] for b in bows], dtype=float)
    centroid = mats.mean(axis=0)
    norms = np.linalg.norm(mats, axis=1) * (np.linalg.norm(centroid) + 1e-9)
    sims = (mats.dot(centroid) / np.maximum(norms, 1e-9)).tolist()
    return [float(max(0.0, s)) for s in sims]

# ---- Weight rules ------------------------------------------------------
def rolling_acc(agents, responses, task_type, ctx) -> List[float]:
    return [max(1e-3, float(getattr(a, "reliability", 0.5))) for a in agents]

def self_conf(agents, responses, task_type, ctx) -> List[float]:
    cal: ConfidenceCalibrator = ctx.get("conf_calibrator")  # may be None
    out = []
    for r in responses:
        p = parse_confidence_pct(r)
        out.append(cal.apply(p) if cal else p)
    # normalize
    s = sum(out) + 1e-9
    return [max(1e-3, v/s) for v in out]

def verifier_bonus(agents, responses, task_type, ctx) -> List[float]:
    """
    ctx should include 'verifier': callable(response)->bool
    weight = 1 + 0.5 if passes else 1.0
    """
    vf = ctx.get("verifier")
    if vf is None:
        return [1.0]*len(responses)
    base = [1.0 + (0.5 if vf(r) else 0.0) for r in responses]
    s = sum(base) + 1e-9
    return [v/s for v in base]

def peer_penalty(agents, responses, task_type, ctx, gamma: float = 0.15) -> List[float]:
    sims = similarity_penalty(responses)  # higher = more similar (worse)
    # turn into (1 - gamma*s)
    raw = [max(0.01, 1.0 - gamma*float(s)) for s in sims]
    s = sum(raw) + 1e-9
    return [v/s for v in raw]

def ensemble(agents, responses, task_type, ctx) -> List[float]:
    """
    w = α*acc + β*conf + γ*verifier - δ*similarity
    α,β,γ,δ can be passed in ctx; defaults: 0.6, 0.2, 0.2, 0.15
    """
    alpha = float(ctx.get("alpha", 0.6))
    beta  = float(ctx.get("beta",  0.2))
    gamma = float(ctx.get("gamma", 0.2))
    delta = float(ctx.get("delta", 0.15))

    acc = np.array(rolling_acc(agents, responses, task_type, ctx))
    conf= np.array(self_conf(agents, responses, task_type, ctx))
    ver = np.array(verifier_bonus(agents, responses, task_type, ctx))
    pen = np.array(peer_penalty(agents, responses, task_type, ctx))

    # combine (note: pen already normalized; treat as subtractive)
    w = alpha*acc + beta*conf + gamma*ver - delta*pen
    w = np.clip(w, 1e-3, None)
    w = w / (w.sum() + 1e-9)
    return w.tolist()

# Registry
REGISTRY = {
    "rolling_acc": rolling_acc,
    "self_conf":   self_conf,
    "verifier":    verifier_bonus,
    "peer":        peer_penalty,
    "ensemble":    ensemble,
}
