#!/bin/bash
# =============================================================================
# AD-SAM Smoke Test: Run every experiment type with minimal data to catch errors
# Each test: 2 train images, 2 val images, 1 epoch, batch_size 2 (= 1 batch each)
# Expected total time: ~5 minutes
# =============================================================================

set +e  # Continue on failure so we can see ALL errors

GPU=0
FAILED=0
PASS_COUNT=0

# Minimal settings: 2 train samples, 2 val samples, 1 batch
COMMON="--max_samples 2 --max_val_samples 2 --num_epochs 1 --batch_size 2 --gpu $GPU"

smoke() {
    local name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "SMOKE TEST: $name"
    echo "$(date)"
    echo "=========================================="
    if "$@"; then
        echo "SMOKE TEST PASSED: $name"
        ((PASS_COUNT++))
    else
        echo "SMOKE TEST FAILED: $name"
        FAILED=1
    fi
}

# AD-SAM variants
smoke "AD-SAM Cityscapes (full)" \
    python train.py --dataset_name cityscapes $COMMON --vis_every 0

smoke "AD-SAM BDD100K (full)" \
    python train.py --dataset_name bdd100k $COMMON --vis_every 0

smoke "AD-SAM ablation: no_deform" \
    python train.py --dataset_name cityscapes $COMMON --ablation no_deform --vis_every 0

smoke "AD-SAM ablation: no_attention" \
    python train.py --dataset_name cityscapes $COMMON --ablation no_attention --vis_every 0

smoke "AD-SAM ablation: sam_encoder_only" \
    python train.py --dataset_name cityscapes $COMMON --ablation sam_encoder_only --vis_every 0

smoke "AD-SAM ablation: ce_loss" \
    python train.py --dataset_name cityscapes $COMMON --ablation ce_loss --vis_every 0

# DeepLabV3 baseline
smoke "DeepLabV3 Cityscapes" \
    python train_deeplabv3.py --dataset_name cityscapes $COMMON

# Efficiency metrics
smoke "Compute Efficiency" \
    python compute_efficiency.py --gpu $GPU

echo ""
echo "=========================================="
echo "SMOKE TEST SUMMARY"
echo "=========================================="
echo "Passed: $PASS_COUNT"
if [ $FAILED -eq 1 ]; then
    echo "STATUS: SOME TESTS FAILED — check output above"
    exit 1
else
    echo "STATUS: ALL SMOKE TESTS PASSED"
    exit 0
fi
