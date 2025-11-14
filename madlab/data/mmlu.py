from typing import List, Dict, Any, Optional

def load_mmlu_examples(limit: Optional[int] = None, split: str = "validation") -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split=split)
        total = len(ds)
        limit = total if (limit is None or limit > total) else limit
        rows = []
        for example in ds.select(range(limit)):
            choices = example.get("choices")
            if not isinstance(choices, list) or len(choices) < 4: continue
            ans_idx = example.get("answer", 0)
            rows.append({
                "question": example.get("question", ""),
                "A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3],
                "answer": ["A","B","C","D"][ans_idx],
            })
        return rows
    except Exception:
        return [{
            "question": "Which structure stores genetic information?",
            "A":"Ribosome","B":"Mitochondrion","C":"Nucleus","D":"Lysosome","answer":"C"
        }][:limit] if limit else []
