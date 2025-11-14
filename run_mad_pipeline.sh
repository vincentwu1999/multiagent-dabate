#!/bin/bash
#SBATCH -D /shared/home/kw459
#SBATCH --account=kamaleswaranlab
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --job-name=mac_pipeline
#SBATCH --output=intelliAgents/mac_pipeline.out
#SBATCH --error=intelliAgents/mac_pipeline.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Ensure log directory exists
mkdir -p intelliAgents

# Activate venv (with SQLite shim & /data caches)
source /data/irb/surgery/pro00114885/kw459/venvs/oca/bin/activate

# (Optional) modules / env
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv  # or: source ~/venv/bin/activate

# Keep threads in check
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Run with srun for proper Slurm integration
python -u /home/kw459/intelliAgents/mad_pipeline.py