
# 🧭 MADLab Tutorial – Multi-Agent Debate Framework

Welcome to **MADLab**, a modular platform for running **multi-agent debates** between LLMs such as GPT-3.5 and GPT-4o-Mini.  
This tutorial will guide you step-by-step through setting up, running, and analyzing experiments.

---

## 1. Getting Started

### 🔧 Requirements
You’ll need:
- Python ≥ 3.9  
- An OpenAI API key (`export OPENAI_API_KEY="sk-..."`)  
- Basic packages:
  ```bash
  pip install openai pandas numpy regex matplotlib datasets
  ```

Optional (for plotting and dataset loading):
```bash
pip install seaborn scikit-learn
```

### 📦 Project Setup
If you cloned the repository:
```bash
git clone https://github.com/<yourname>/madlab.git
cd madlab
export PYTHONPATH=.
```

You can also install it in editable mode:
```bash
pip install -e .
```

---

## 2. Running Your First Experiment

The main entrypoint is `madlab.cli`.

### Example 1 – GSM8K math reasoning
```bash
python -m madlab.cli --tasks gsm8k --limit 50   --weight-rule ensemble --alpha 0.6 --beta 0.2 --gamma 0.2 --delta 0.15
```

This command:
- Runs 50 GSM8K math questions  
- Creates 3 agents that debate for 2 rounds  
- Uses **ensemble weighted voting** to decide the final answer  
- Saves all results under `outputs/gsm8k_results.csv`

### Example 2 – Disable weighted voting
```bash
python -m madlab.cli --tasks mmlu --limit 50 --no-weighted
```

This uses simple majority voting instead of reliability-based weighting.

---

## 3. Exploring the Results

After the run, open `outputs/gsm8k_results.csv` in any CSV viewer.  
Each row corresponds to a question and includes:

| Column | Meaning |
|--------|----------|
| `single`, `reflection`, `majority`, `debate_weighted` | 0/1 accuracy |
| `*_text` | Raw text responses from agents |
| `agent_final_texts` | All final agent messages (JSON-encoded) |
| `debate_history` | Full multi-round conversation (JSON) |

You’ll also find:
- `outputs/summary.csv` – average scores per method  
- Optional plots (barplots per task) if `--no-plots` was not used  

---

## 4. Understanding Weighted Voting

Weighted voting lets better or more confident agents have greater influence.  
In MADLab, each agent’s weight w_i is computed as:

w_i = α * reliability_i + β * confidence_i + γ * verifier_i - δ * similarity_i

**Signals:**
- **Reliability**: running accuracy (EMA) from calibration items  
- **Confidence**: parsed from “Confidence: 83%” in outputs  
- **Verifier**: 1 if passes programmatic correctness checks  
- **Similarity penalty**: lowers weight for copycat responses  

CLI examples:
```bash
# Custom weighting mix
python -m madlab.cli --tasks gsm8k --weight-rule ensemble   --alpha 0.5 --beta 0.25 --gamma 0.25 --delta 0.1

# Rolling accuracy only
python -m madlab.cli --tasks gsm8k --weight-rule rolling_acc
```

To disable weighting:
```bash
--no-weighted
```

---

## 5. Running on HPC (Duke DCC example)

Save this as `run_mad_gpt3.5.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=mad_gpt3.5
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --partition=scavenger
#SBATCH --account=jilab
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate py3.12

export OPENAI_API_KEY="sk-xxxx"
export MAD_MODEL="gpt-3.5-turbo"
export MAD_TEMPERATURE="0.5"

PYTHONPATH=. python -m madlab.cli --tasks gsm8k,mmlu --limit 100   --weight-rule ensemble --alpha 0.6 --beta 0.2 --gamma 0.2 --delta 0.15
```

Submit it:
```bash
sbatch run_mad_gpt3.5.slurm
```

---

## 6. Using the New BioASQ Dataset

BioASQ is a biomedical QA benchmark with **factoid**, **yes/no**, and **list** questions.  
It’s fully integrated into MADLab.

### 6.1 Using HuggingFace Datasets
If `datasets` can access BioASQ subsets:
```bash
python -m madlab.cli --tasks bioasq --bioasq-subset factoid --limit 100
```

### 6.2 Using Local Data
Prepare a file (JSON or JSONL):
```json
{
  "type": "factoid",
  "question": "Which gene is responsible for cystic fibrosis?",
  "answers": ["CFTR"]
}
```
Then run:
```bash
export BIOASQ_PATH=/path/to/bioasq_data
python -m madlab.cli --tasks bioasq --bioasq-subset all --limit 50
```

The system will automatically use your local file if no HF dataset is available.

---

## 7. Inspecting BioASQ Results

Outputs:
- `outputs/bioasq_results.csv` – per-question results  
- Columns include `type`, `question`, `gold_answers`, and each agent’s output  
- Evaluation metrics:
  - **Factoid / Yes-No:** Exact match accuracy  
  - **List:** F1 overlap between predicted and gold entities  

---

## 8. Advanced Experiments

### 🧠 Self-Improved Debates
Use debate-generated examples as pseudo-labels to fine-tune or retrain your agents.

### ⚖️ Weighted Voting Ablations
Compare:
```bash
python -m madlab.cli --tasks gsm8k --weight-rule rolling_acc
python -m madlab.cli --tasks gsm8k --weight-rule verifier
python -m madlab.cli --tasks gsm8k --weight-rule ensemble
```

### 💬 Agent Diversity
Edit `madlab/agents.py` to change personas:
- “Skeptical Mathematician”
- “Analytical Researcher”
- “Creative Thinker”
  
Each one brings different reasoning biases to the debate.

---

## 9. Troubleshooting

| Issue | Cause / Fix |
|-------|--------------|
| `KeyError: 'answer'` | Mismatch in prompt template; make sure `PROMPTS[...]` keys match runner calls |
| Nothing happens | Add `if __name__ == "__main__": main()` to `cli.py` |
| API rate limits | Reduce `--limit` or add delays in `llm.py` |
| Empty CSV | Check your API key and model name (`MAD_MODEL`, `MAD_TEMPERATURE`) |

---

## 10. Where to Go Next

- Explore **BioASQ + GSM8K mixed debates** for cross-domain generalization  
- Try **multi-round reflection** before debate  
- Add **retrieval augmentation** (e.g., PubMed abstracts for BioASQ)  
- Tune weighting coefficients for different domains  

---

**Congratulations 🎉** — you can now:
- Run single-agent, reflection, and debate-based reasoning
- Evaluate debates with weighted or unweighted voting
- Extend to new datasets like BioASQ or MedQA  
- Analyze detailed debate logs for each question  
