#!/usr/bin/env bash
# Reproduce all results from the paper.
# Assumes datasets have been downloaded to data/raw/ (see docs/DATASETS.md).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1/4: Illustrative simulation (Section 5.1-5.2) ==="
python scripts/run_benchmark.py \
  --dataset simulation \
  --n-seeds 10 \
  --output-dir results/simulation

if [ -d "data/raw/ml-1m" ]; then
  echo ""
  echo "=== 2/4: MovieLens 1M replay (Section 5.3) ==="
  python scripts/run_benchmark.py \
    --dataset movielens \
    --data-dir data/raw/ml-1m \
    --output-dir results/movielens
else
  echo "SKIP: data/raw/ml-1m not found. See docs/DATASETS.md"
fi

if [ -d "data/raw/retailrocket" ]; then
  echo ""
  echo "=== 3/4: RetailRocket replay (Section 5.4) ==="
  python scripts/run_benchmark.py \
    --dataset retailrocket \
    --data-dir data/raw/retailrocket \
    --output-dir results/retailrocket
else
  echo "SKIP: data/raw/retailrocket not found. See docs/DATASETS.md"
fi

if [ -d "data/raw/rees46" ]; then
  echo ""
  echo "=== 4/4: REES46 Cosmetics replay (Section 5.5) ==="
  python scripts/run_benchmark.py \
    --dataset rees46 \
    --data-dir data/raw/rees46 \
    --output-dir results/rees46
else
  echo "SKIP: data/raw/rees46 not found. See docs/DATASETS.md"
fi

echo ""
echo "=== All benchmarks complete. Outputs in results/ ==="
