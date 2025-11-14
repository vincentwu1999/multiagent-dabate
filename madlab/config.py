import os, pathlib

# Reproducibility & defaults
SEED = int(os.environ.get("MAD_SEED", "7"))
DEFAULT_MODEL = os.environ.get("MAD_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.environ.get("MAD_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("MAD_MAX_TOKENS", "512"))
N_AGENTS = int(os.environ.get("MAD_AGENTS", "3"))
N_ROUNDS = int(os.environ.get("MAD_ROUNDS", "2"))

# Paths
OUT_DIR = pathlib.Path(os.environ.get("MAD_OUT", "./outputs")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_out_dir(path: str):
    p = pathlib.Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
