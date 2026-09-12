#!/bin/bash
# Every entry point on 2 training / 2 validation images for 1 epoch (~10 min on one GPU).
# Usage: bash scripts/smoke_test.sh [--embedding_dir embeddings]
set +e
cd "$(dirname "$0")/.."
GPU=${GPU:-0}; EMB=${1:-}; PASS=0; FAIL=0
COMMON="--max_samples 2 --max_val_samples 2 --num_epochs 1 --batch_size 2 --gpu $GPU --output_dir results/smoke"
run() { echo; echo "== $1"; shift; if "$@"; then echo "PASSED"; ((PASS++)); else echo "FAILED"; FAIL=1; fi; }
if [ -n "$EMB" ]; then
  run "AD-SAM cached + flip"          python train.py adsam $COMMON $EMB --flip_augment --vis_every 1
  run "AD-SAM cached + eval_native"   python train.py adsam $COMMON $EMB --eval_native
  for ab in no_deform no_attention sam_encoder_only ce_loss; do
    run "AD-SAM ablation $ab"          python train.py adsam $COMMON $EMB --ablation $ab --vis_every 0
  done
fi
run "AD-SAM live encoder"             python train.py adsam $COMMON --vis_every 0
run "DeepLabV3 + flip"                python train.py deeplabv3 $COMMON --flip_augment
run "DeepLabV3 + eval_native"         python train.py deeplabv3 $COMMON --eval_native
run "efficiency"                      python scripts/compute_efficiency.py --gpu $GPU
echo; echo "passed $PASS"; [ $FAIL -eq 0 ] && echo "ALL SMOKE TESTS PASSED" || { echo "SOME TESTS FAILED"; exit 1; }
