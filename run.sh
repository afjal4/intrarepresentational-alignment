#!/bin/bash
#SBATCH --job-name=intra-align
#SBATCH --cpus-per-task=8          # gensim/numpy parallelism
#SBATCH --mem=48G                  # 6 embedding models including Numberbatch
#SBATCH --time=08:00:00            # ~4-8 h for 6 models x 75 pairs x 500 perms
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/%u/intra-align/logs/%j.out
#SBATCH --error=/vol/bitbucket/%u/intra-align/logs/%j.err

set -euo pipefail

# ── virtualenv ────────────────────────────────────────────────────────────────
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate

# ── gensim model cache ────────────────────────────────────────────────────────
# Models are ~4 GB total; keep them in bitbucket so they survive between jobs.
export GENSIM_DATA_HOME="/vol/bitbucket/${USER}/gensim-data"
mkdir -p "${GENSIM_DATA_HOME}"

# ── working directory ─────────────────────────────────────────────────────────
REPO="/vol/bitbucket/${USER}/intra-align"
cd "${REPO}"
mkdir -p logs figures

# ── run ───────────────────────────────────────────────────────────────────────
echo "Starting at $(date)"
echo "Node: $(hostname)  |  Python: $(python --version)"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=28800 \
    --ExecutePreprocessor.kernel_name=python3 \
    main.ipynb \
    --output main_executed.ipynb

echo "Done at $(date)"
echo "Figures:"
ls -lh figures/*.png 2>/dev/null || echo "  (none found)"
