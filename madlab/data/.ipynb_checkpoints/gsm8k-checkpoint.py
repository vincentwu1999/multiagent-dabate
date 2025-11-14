from typing import List, Dict, Any, Optional

def load_gsm8k_examples(limit: Optional[int] = None, split: str = "test") -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        import regex as re
        ds = load_dataset("openai/gsm8k", "main", split=split)
        total = len(ds)
        limit = total if (limit is None or limit > total) else limit
        rows = []
        for example in ds.select(range(limit)):
            question = example["question"]
            answer_text = example["answer"]
            m = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_text)
            gold = float(m.group(1)) if m else None
            if gold is not None and float(gold).is_integer(): gold = int(gold)
            rows.append({"problem": question, "answer": gold})
        return rows
    except Exception:
        fb = [
            {"problem": "Regina wrote 9 novels last year. If this is 3/4 of this year's output, how many this year?", "answer": 12},
            {"problem": "A $12 toy with $3 coupon and 25% off after coupon – final price?", "answer": 6.75},
        ]
        return fb[:limit] if limit else fb
