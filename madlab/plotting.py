import os, numpy as np, matplotlib.pyplot as plt

def barplot_results(task_name, df, is_accuracy=True, out_dir="./outputs"):
    metrics = ["single", "reflection", "majority", "debate_weighted"]
    labels  = ["Single Agent", "Reflection", "Multi-Agent (Majority)", "Multi-Agent (Debate)"]
    vals = [100.0 * df[m].mean() for m in metrics]
    plt.figure(figsize=(6,4))
    xs = np.arange(len(metrics))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=15, ha="right")
    plt.ylabel("Accuracy (%)" if is_accuracy else "Factuality (%)")
    plt.title(task_name)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{task_name.lower().replace(' ','_')}_bars.png")
    plt.savefig(path, dpi=180)
    plt.close(); print(f"[Saved] {path}")

def ablation_plot(title, xvals, yvals, xlabel, ylabel, fname, out_dir="./outputs"):
    plt.figure(figsize=(5,4))
    plt.plot(xvals, yvals, marker="o")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=180)
    plt.close(); print(f"[Saved] {path}")
