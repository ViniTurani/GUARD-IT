#!/usr/bin/env bash
# Reproduce the GUARD-IT TOFU rows: forget01 / forget05, Llama-3.2-1B / 3.2-3B / 3.1-8B, seeds 0-4.
# Requires benchmarks/open-unlearning (see README, "Evaluation").
#
#   bash run_all.sh                         # everything
#   SEEDS="0" SPLITS="forget01" bash run_all.sh
#
# Results: results/tofu/<model>_<split>_s<seed>/ ; summary: results/tofu/summary_groups.csv
set -u
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"0 1 2 3 4"}
SPLITS=${SPLITS:-"forget01 forget05"}
MODELS=${MODELS:-"1b 3b 8b"}
OUT=${OUT:-results/tofu}
SV=${SV:-steering_vectors}
mkdir -p "$OUT"

for m in $MODELS; do
  case $m in
    1b) hf=open-unlearning/tofu_Llama-3.2-1B-Instruct_full; layer=4; alpha=0.2; date="28 Apr 2026" ;;
    3b) hf=open-unlearning/tofu_Llama-3.2-3B-Instruct_full; layer=7; alpha=0.4; date="28 Apr 2026" ;;
    8b) hf=open-unlearning/tofu_Llama-3.1-8B-Instruct_full; layer=8; alpha=0.5; date="26 Jul 2024" ;;
    *)  echo "unknown model $m"; exit 1 ;;
  esac
  for split in $SPLITS; do
    for seed in $SEEDS; do
      run=$OUT/${m}_${split}_s$seed
      export GUARD_TEMPLATE_DATE="$date"   # "Today Date" in the chat template; the gate reads it
      guard cluster configs/examples/tofu_${m}_$split.yaml \
        --override output_dir="$SV/" --override seed=$seed > "$run.cluster.log" 2>&1 \
        || { echo "FAIL cluster $m $split s$seed (see $run.cluster.log)"; continue; }
      svdir=$(ls -td "$SV"/"${hf//\//_}"/orthogonal_clustered/${split}_Kf*_s$seed | head -1)
      guard benchmark tofu benchmarks/configs/eval_tofu_$split.yaml \
        --sv-dir "$svdir/residual/layer_$layer" --coeffs $alpha \
        --override model=$hf --override layer=$layer \
        --output-path "$run/results.json" > "$run.eval.log" 2>&1 \
        && echo "OK   $m $split s$seed" || echo "FAIL eval $m $split s$seed (see $run.eval.log)"
    done
  done
done

python benchmarks/recompute_mem.py "$OUT" --out "$OUT/summary" \
  --path-regex '(?P<run>[^/]+)_s(?P<seed>\d+)/logs/[^/]+/(?P<coeff>[^/]+)/TOFU_EVAL'
