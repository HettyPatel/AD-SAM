#!/bin/bash
# Reproduce the paper's main table: AD-SAM and DeepLabV3 at 100 / 500 / 1000 / all Cityscapes images and all of BDD100K.
# Usage:  bash scripts/run_experiments.sh                 # everything, in order
#         bash scripts/run_experiments.sh adsam_cs_100 deeplab_cs_100
# Env:    GPU=0  EPOCHS=100  BS=2  EMB=embeddings  EXTRA="--eval_native --seed 0"
set -o pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}; EPOCHS=${EPOCHS:-100}; BS=${BS:-2}; EMB=${EMB:-embeddings}; EXTRA=${EXTRA:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs; LOG=results/logs/experiments_$(date +%Y%m%d_%H%M%S).log
ALL="adsam_cs_100 deeplab_cs_100 adsam_cs_500 deeplab_cs_500 adsam_cs_1000 deeplab_cs_1000 adsam_cs_full deeplab_cs_full adsam_bdd_full deeplab_bdd_full"
adsam()   { python train.py adsam     --dataset_name "$1" --max_samples "$2" --num_epochs $EPOCHS --batch_size $BS --gpu $GPU --embedding_dir $EMB --flip_augment $EXTRA; }
deeplab() { python train.py deeplabv3 --dataset_name "$1" --max_samples "$2" --num_epochs $EPOCHS --batch_size $BS --gpu $GPU --flip_augment $EXTRA; }
{
echo "=== start $(date)  GPU=$GPU  git $(git rev-parse --short HEAD 2>/dev/null)"
for r in ${@:-$ALL}; do
  echo; echo "### $r  $(date)"
  case $r in
    adsam_cs_100) adsam cityscapes 100;;     deeplab_cs_100) deeplab cityscapes 100;;
    adsam_cs_500) adsam cityscapes 500;;     deeplab_cs_500) deeplab cityscapes 500;;
    adsam_cs_1000) adsam cityscapes 1000;;   deeplab_cs_1000) deeplab cityscapes 1000;;
    adsam_cs_full) adsam cityscapes None;;   deeplab_cs_full) deeplab cityscapes None;;
    adsam_bdd_full) adsam bdd100k None;;     deeplab_bdd_full) deeplab bdd100k None;;
    *) echo "unknown run: $r";;
  esac
  echo "### $r exit $?"
done
echo; echo "=== done $(date)"; cut -d, -f1-18 results/results_summary.csv
} 2>&1 | tee "$LOG"
