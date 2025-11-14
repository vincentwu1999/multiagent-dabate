from typing import List
from ..agents import make_agents
from ..debate import Debate
from ..prompts import PROMPTS
from ..data.arithmetic import gen_arithmetic
from ..data.gsm8k import load_gsm8k_examples
from ..evals import eval_arithmetic_item, eval_gsm8k_item

def ablation_agents_gsm8k(llm, limit: int, out_dir: str, plotter):
    xvals, accs = [], []
    for k in [1, 2, 3, 4, 5]:
        ags = make_agents(llm, n_agents=k, persona_diversity=True)
        data = load_gsm8k_examples(limit=limit)
        ok_list = []
        for itm in data:
            prob, gold = itm["problem"], itm["answer"]
            sp = PROMPTS["gsm8k_start"].format(problem=prob)
            tmpl = PROMPTS["gsm8k_debate"].replace("{problem}", prob)
            db = Debate(agents=ags, task_type="gsm8k", start_prompt=sp, debate_prompt_tmpl=tmpl, rounds=2).run()
            ok_list.append(eval_gsm8k_item(gold, db["final_weighted"]))
        xvals.append(k); accs.append(100.0 * (sum(ok_list) / len(ok_list)))
    plotter("Performance vs Number of Debating Agents", xvals, accs, "Agents", "GSM8K Accuracy (%)", "ablation_agents_gsm8k.png", out_dir)

def ablation_rounds_arithmetic(llm, limit: int, out_dir: str, plotter):
    xvals, accs = [], []
    items = gen_arithmetic(limit)
    for rds in [1, 2, 3, 4]:
        ags = make_agents(llm, n_agents=3, persona_diversity=False)
        ok_list: List[bool] = []
        for itm in items:
            expr = itm["expr"]
            import regex as re
            start = PROMPTS["arithmetic_start"].format(*re.findall(r"\d+", expr))
            db = Debate(agents=ags, task_type="arithmetic", start_prompt=start,
                        debate_prompt_tmpl=PROMPTS["arithmetic_debate"], rounds=rds).run()
            ok_list.append(eval_arithmetic_item(expr, db["final_weighted"]))
        xvals.append(rds); accs.append(100.0 * (sum(ok_list) / len(ok_list)))
    plotter("Math Accuracy vs Debate Rounds", xvals, accs, "Debate Rounds", "Arithmetic Accuracy (%)", "ablation_rounds_arith.png", out_dir)
