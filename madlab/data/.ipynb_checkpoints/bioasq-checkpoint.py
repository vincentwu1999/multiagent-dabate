# madlab/cli.py
import argparse, json
import pandas as pd

from .config import set_out_dir
from .llm import LLMClient

from .runners.core import (
    run_task_arithmetic, run_task_gsm8k, run_task_mmlu,
    run_task_chess_validity, run_task_chess_move,
    run_task_biographies
)

def main():
    p = argparse.ArgumentParser(description="madlab – Modular Multi-Agent Debate")
    p.add_argument("--out", type=str, default="./outputs", help="Output directory")
    p.add_argument("--limit", type=int, default=50, help="Max items per task")
    p.add_argument("--tasks", type=str, default="arith,gsm8k,mmlu,chess_valid,chess_move,bio",
                   help="Comma-separated list: arith,gsm8k,mmlu,chess_valid,chess_move,bio")
    p.add_argument("--no-plots", action="store_true", help="Disable plotting")
    p.add_argument("--model", type=str, default=None, help="Override MAD_MODEL")
    p.add_argument("--temperature", type=float, default=None, help="Override MAD_TEMPERATURE")

    # weighted-vote controls
    p.add_argument("--weight-rule", type=str, default="ensemble",
                   help="rolling_acc | self_conf | verifier | peer | ensemble")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta",  type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--no-weighted", action="store_true",
                   help="Disable weighted voting (use majority instead)")

    args = p.parse_args()

    out_dir = str(set_out_dir(args.out))
    llm = LLMClient(model=args.model, temperature=args.temperature) if (args.model or args.temperature is not None) else LLMClient()

    task_map = {
        "arith":      (run_task_arithmetic, True),
        "gsm8k":      (run_task_gsm8k, True),
        "mmlu":       (run_task_mmlu, True),
        "chess_valid":(run_task_chess_validity, True),
        "chess_move": (run_task_chess_move, True),
        "bio":        (run_task_biographies, False),
        # legacy alias (optional):
        "chess":      (run_task_chess_validity, True),
    }

    summaries = []
    tasks = [x.strip() for x in args.tasks.split(',') if x.strip()]
    for key in tasks:
        if key not in task_map:
            print(f"[WARN] Unknown task: {key}")
            continue
        runner, is_acc = task_map[key]
        res = runner(
            llm,
            n_items=args.limit,
            use_debate=True,
            persona_diversity=(key in {"bio", "chess_move"}),  # tweak as you like
            use_weighted=(not args.no_weighted),
            weight_rule=args.weight_rule,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta
        )
        df = res["df"]

        # CSV output only
        csv_path = f"{out_dir}/{key}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")

        # Optional plots (best-effort)
        if not args.no_plots and not df.empty:
            try:
                from .plotting import barplot_results
                title = key.upper() if key not in ("bio",) else "BIOGRAPHIES FACTUALITY"
                barplot_results(title, df, is_accuracy=is_acc, out_dir=out_dir)
            except Exception as e:
                print(f"[WARN] Plotting skipped: {e}")

        summaries.append({
            "task": key,
            "Single": round(df["single"].mean()*100.0, 2) if not df.empty else 0.0,
            "Reflection": round(df["reflection"].mean()*100.0, 2) if not df.empty else 0.0,
            "Majority": round(df["majority"].mean()*100.0, 2) if not df.empty else 0.0,
            "DebateWeighted": round(df["debate_weighted"].mean()*100.0, 2) if not df.empty else 0.0,
        })

    with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    try:
        pd.DataFrame(summaries).to_csv(f"{out_dir}/summary.csv", index=False)
        print(f"[Saved] {out_dir}/summary.csv")
    except Exception as e:
        print(f"[WARN] Could not save summary.csv: {e}")

    print("\n=== Summary (%) ===")
    for s in summaries:
        print(s)

if __name__ == "__main__":
    main()
