#!/bin/bash
# =============================================================================
# AD-SAM Full Overnight Experiment Suite
#
# Runs all experiments needed for the revised paper:
#   - Data efficiency (AD-SAM at different training sizes)
#   - DeepLabV3 baseline at same sizes
#   - Ablation study (5 variants)
#   - CRF post-processing
#   - Efficiency metrics
#   - Results collection
#
# Usage:
#   nohup bash run_overnight.sh > overnight_output.log 2>&1 &
#   tail -f overnight_output.log       # monitor progress
#   tail -f results/experiment_status.log  # check pass/fail
#
# Expected runtime: ~36-48 hours on single A6000 (batch_size=2)
# =============================================================================

GPU=0
BATCH_SIZE=2     # batch size 2 to leave GPU headroom for others
EPOCHS=100
VIS_EVERY=25     # visualize every 25 epochs to save time/disk

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Results directory
mkdir -p results/logs
mkdir -p results/checkpoints
mkdir -p results/efficiency

# Status log
STATUS_LOG="results/experiment_status.log"
echo "=== AD-SAM Overnight Experiments ===" > $STATUS_LOG
echo "Started: $(date)" >> $STATUS_LOG
echo "" >> $STATUS_LOG

# ── Helper function ──────────────────────────────────────────────────────────

run_experiment() {
    local exp_name="$1"
    local exp_num="$2"
    shift 2
    local cmd="$@"

    echo ""
    echo "================================================================"
    echo "[Exp $exp_num] $exp_name"
    echo "$(date)"
    echo "Command: $cmd"
    echo "================================================================"

    echo "$(date) | [Exp $exp_num] STARTED  | $exp_name" >> $STATUS_LOG

    if eval $cmd; then
        echo "$(date) | [Exp $exp_num] COMPLETED | $exp_name" >> $STATUS_LOG
    else
        echo "$(date) | [Exp $exp_num] FAILED    | $exp_name" >> $STATUS_LOG
    fi
}

# ── GROUP 1: AD-SAM Data Efficiency ─────────────────────────────────────────

echo ""
echo "############################################################"
echo "# GROUP 1: AD-SAM Data Efficiency                          #"
echo "############################################################"

run_experiment "AD-SAM Cityscapes 100 samples" 1 \
    "python train.py --dataset_name cityscapes --max_samples 100 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM Cityscapes 500 samples" 2 \
    "python train.py --dataset_name cityscapes --max_samples 500 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM Cityscapes 1000 samples" 3 \
    "python train.py --dataset_name cityscapes --max_samples 1000 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM Cityscapes FULL (2975)" 4 \
    "python train.py --dataset_name cityscapes --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM BDD100K 100 samples" 5 \
    "python train.py --dataset_name bdd100k --max_samples 100 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM BDD100K 500 samples" 6 \
    "python train.py --dataset_name bdd100k --max_samples 500 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM BDD100K 1000 samples" 7 \
    "python train.py --dataset_name bdd100k --max_samples 1000 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "AD-SAM BDD100K FULL (5968)" 8 \
    "python train.py --dataset_name bdd100k --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --gpu $GPU --vis_every $VIS_EVERY"


# ── GROUP 2: DeepLabV3 Baseline ──────────────────────────────────────────────

echo ""
echo "############################################################"
echo "# GROUP 2: DeepLabV3 Baseline (fair comparison)            #"
echo "############################################################"

run_experiment "DeepLabV3 Cityscapes 100 samples" 9 \
    "python train_deeplabv3.py --dataset_name cityscapes --max_samples 100 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU"

run_experiment "DeepLabV3 Cityscapes 500 samples" 10 \
    "python train_deeplabv3.py --dataset_name cityscapes --max_samples 500 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU"

run_experiment "DeepLabV3 Cityscapes 1000 samples" 11 \
    "python train_deeplabv3.py --dataset_name cityscapes --max_samples 1000 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU"

run_experiment "DeepLabV3 Cityscapes FULL" 12 \
    "python train_deeplabv3.py --dataset_name cityscapes --num_epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU"

run_experiment "DeepLabV3 BDD100K 100 samples" 13 \
    "python train_deeplabv3.py --dataset_name bdd100k --max_samples 100 --num_epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU"

run_experiment "DeepLabV3 BDD100K FULL" 14 \
    "python train_deeplabv3.py --dataset_name bdd100k --num_epochs $EPOCHS --batch_size $BATCH_SIZE --gpu $GPU"


# ── GROUP 3: Ablation Study ─────────────────────────────────────────────────
# All on Cityscapes full dataset, 100 epochs
# Note: "full" ablation = same as Exp #4, skip if already ran

echo ""
echo "############################################################"
echo "# GROUP 3: Ablation Study (Cityscapes full)                #"
echo "############################################################"

# Exp 4 (full) already covers the baseline — skip #15

run_experiment "Ablation: no_deform (Cityscapes full)" 16 \
    "python train.py --dataset_name cityscapes --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --ablation no_deform --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "Ablation: no_attention (Cityscapes full)" 17 \
    "python train.py --dataset_name cityscapes --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --ablation no_attention --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "Ablation: sam_encoder_only (Cityscapes full)" 18 \
    "python train.py --dataset_name cityscapes --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --ablation sam_encoder_only --gpu $GPU --vis_every $VIS_EVERY"

run_experiment "Ablation: ce_loss (Cityscapes full)" 19 \
    "python train.py --dataset_name cityscapes --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --ablation ce_loss --gpu $GPU --vis_every $VIS_EVERY"


# ── GROUP 4: CRF Post-Processing ────────────────────────────────────────────

echo ""
echo "############################################################"
echo "# GROUP 4: CRF Post-Processing                            #"
echo "############################################################"

run_experiment "AD-SAM Cityscapes FULL + CRF" 20 \
    "python train.py --dataset_name cityscapes --max_samples None --num_epochs $EPOCHS --batch_size $BATCH_SIZE --loss hybrid --apply_crf --gpu $GPU --vis_every $VIS_EVERY"


# ── GROUP 5: Efficiency Metrics ──────────────────────────────────────────────

echo ""
echo "############################################################"
echo "# GROUP 5: Efficiency Metrics                              #"
echo "############################################################"

run_experiment "Compute Efficiency (params/FLOPs/latency)" 21 \
    "python compute_efficiency.py --gpu $GPU"


# ── COLLECT RESULTS ──────────────────────────────────────────────────────────

echo ""
echo "############################################################"
echo "# COLLECTING ALL RESULTS                                   #"
echo "############################################################"

python collect_results.py


# ── ORGANIZE OUTPUT FILES ────────────────────────────────────────────────────

echo ""
echo "Organizing output files..."

# Move logs
mv -f training_log_*.txt results/logs/ 2>/dev/null || true

# Move checkpoints
mv -f dual_encoder_*.pth results/checkpoints/ 2>/dev/null || true
mv -f deeplabv3_*.pth results/checkpoints/ 2>/dev/null || true

echo ""
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Results summary:  results/results_summary.csv"
echo "Efficiency:       results/efficiency/efficiency_results.csv"
echo "Experiment log:   results/experiment_status.log"
echo "Training logs:    results/logs/"
echo "Checkpoints:      results/checkpoints/"
echo ""

echo "$(date) | ALL EXPERIMENTS FINISHED" >> $STATUS_LOG
