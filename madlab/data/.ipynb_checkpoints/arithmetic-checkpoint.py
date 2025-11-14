import random
from typing import List, Dict, Any
from ..utils import safe_eval_arithmetic

random.seed(7)

def gen_arithmetic(n: int = 100) -> List[Dict[str, Any]]:
    items = []
    for _ in range(n):
        xs = [random.randint(0, 30) for __ in range(6)]
        expr = f"{xs[0]}+{xs[1]}*{xs[2]}+{xs[3]}-{xs[4]}*{xs[5]}"
        ans = safe_eval_arithmetic(expr)
        items.append({"expr": expr, "answer": ans})
    return items
