#!/bin/bash
#SBATCH --job-name=intra-align
#SBATCH --partition=cpu            # confirm with: sinfo -s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # gensim/numpy parallelism
#SBATCH --mem=32G                  # 5 embedding models + permutation matrices
#SBATCH --time=04:00:00            # ~2-4 h for 5 models x 75 pairs x 500 perms
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# ── environment ──────────────────────────────────────────────────────────────
# Adjust the module name to match what Imperial's RCS provides.
# Run `module avail python` or `module avail anaconda` on the login node to find it.
module load anaconda3/personal   # or: module load python/3.11.9

# Activate the repo's virtualenv (created locally with: python -m venv venv)
source venv/bin/activate

# ── gensim model cache ────────────────────────────────────────────────────────
# Point gensim at a persistent directory so models survive between jobs.
# Change this to a scratch/RDS path if $HOME quota is tight.
export GENSIM_DATA_HOME="${HOME}/gensim-data"
mkdir -p "${GENSIM_DATA_HOME}"

# ── output dirs ───────────────────────────────────────────────────────────────
mkdir -p logs figures

# ── run ───────────────────────────────────────────────────────────────────────
echo "Starting at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(python --version)"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=python3 \
    main.ipynb \
    --output main_executed.ipynb

echo "Notebook executed successfully at $(date)"

# ── report figures ────────────────────────────────────────────────────────────
echo "Figures written:"
ls -lh figures/*.png 2>/dev/null || echo "  (none found — check notebook output)"
