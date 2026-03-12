"""
Compute efficiency metrics (parameters, FLOPs, inference time, peak memory)
for AD-SAM model variants and DeepLabV3 baseline.

Usage:
    python compute_efficiency.py --gpu 0
"""

import argparse
import time
import csv
import os

import torch
import torch.nn as nn
import torchvision

from utils.setup import initialize_sam_model
from run_ablations import build_ablation_model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_component_params(model, ablation):
    """Break down parameter counts by component."""
    components = {}

    if hasattr(model, 'sam_encoder'):
        components['sam_encoder'] = sum(p.numel() for p in model.sam_encoder.parameters())
    if hasattr(model, 'resnet_encoder'):
        components['resnet_encoder'] = sum(p.numel() for p in model.resnet_encoder.parameters())
    if hasattr(model, 'fusion_layers'):
        components['fusion'] = sum(p.numel() for p in model.fusion_layers.parameters())

    # Decoder: up1-up4 + classifier
    decoder_params = 0
    for i in range(1, 5):
        attr = f'up{i}'
        if hasattr(model, attr):
            decoder_params += sum(p.numel() for p in getattr(model, attr).parameters())
    if hasattr(model, 'classifier'):
        decoder_params += sum(p.numel() for p in model.classifier.parameters())
    components['decoder'] = decoder_params

    return components


def estimate_flops(model, input_size=(1, 3, 1024, 1024), device='cuda'):
    """Rough FLOPs estimate using a forward hook-based counter."""
    flop_count = [0]

    def conv_hook(module, inp, out):
        # FLOPs = 2 * Cin * Cout * Kh * Kw * Hout * Wout / groups
        inp_shape = inp[0].shape
        out_shape = out.shape
        if hasattr(module, 'kernel_size'):
            kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        else:
            kh, kw = 3, 3
        groups = module.groups if hasattr(module, 'groups') else 1
        cin = module.in_channels if hasattr(module, 'in_channels') else inp_shape[1]
        cout = out_shape[1]
        hout, wout = out_shape[2], out_shape[3]
        flops = 2 * cin * cout * kh * kw * hout * wout // groups
        flop_count[0] += flops

    def linear_hook(module, inp, out):
        flops = 2 * module.in_features * module.out_features
        flop_count[0] += flops

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, torchvision.ops.DeformConv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy = torch.randn(input_size, device=device)
    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flop_count[0]


def measure_inference(model, device, input_size=(1, 3, 1024, 1024),
                      warmup=5, repeats=20):
    """Measure average inference time and peak GPU memory."""
    model.eval()
    dummy = torch.randn(input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    torch.cuda.synchronize(device)

    # Reset peak memory
    torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            model(dummy)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    avg_time = elapsed / repeats
    fps = 1.0 / avg_time
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    return avg_time, fps, peak_mem


def build_deeplabv3(num_classes=19):
    """Build DeepLabV3-ResNet101 for comparison."""
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    )
    model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)
    return model


class DeepLabV3Wrapper(nn.Module):
    """Wrapper so DeepLabV3 has the same forward signature (returns tensor)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    os.makedirs("results/efficiency", exist_ok=True)

    sam_model = initialize_sam_model()

    # Models to evaluate
    models_to_eval = {
        "AD-SAM (full)": ("full", None),
        "AD-SAM (no_deform)": ("no_deform", None),
        "AD-SAM (no_attention)": ("no_attention", None),
        "AD-SAM (sam_encoder_only)": ("sam_encoder_only", None),
        "DeepLabV3-ResNet101": (None, "deeplabv3"),
    }

    results = []

    for name, (ablation, baseline) in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        if baseline == "deeplabv3":
            raw_model = build_deeplabv3(num_classes=19)
            model = DeepLabV3Wrapper(raw_model).to(device).float()
            # Freeze nothing for param counting
        else:
            model = build_ablation_model(ablation, sam_model, num_classes=19).to(device).float()
            # Freeze SAM encoder as in training
            for param in model.sam_encoder.parameters():
                param.requires_grad = False

        total_params, trainable_params = count_parameters(model)

        print(f"  Total params:     {total_params / 1e6:.2f}M")
        print(f"  Trainable params: {trainable_params / 1e6:.2f}M")

        # Component breakdown
        if ablation is not None:
            components = count_component_params(model, ablation)
            for comp, cnt in components.items():
                print(f"    {comp}: {cnt / 1e6:.2f}M")

        # FLOPs
        try:
            flops = estimate_flops(model, device=device)
            gflops = flops / 1e9
            print(f"  GFLOPs: {gflops:.2f}")
        except Exception as e:
            print(f"  GFLOPs: error ({e})")
            gflops = -1

        # Inference time
        try:
            avg_time, fps, peak_mem = measure_inference(model, device)
            print(f"  Inference time: {avg_time * 1000:.1f} ms")
            print(f"  FPS: {fps:.2f}")
            print(f"  Peak memory: {peak_mem:.0f} MB")
        except Exception as e:
            print(f"  Inference: error ({e})")
            avg_time, fps, peak_mem = -1, -1, -1

        results.append({
            "model": name,
            "total_params_M": f"{total_params / 1e6:.2f}",
            "trainable_params_M": f"{trainable_params / 1e6:.2f}",
            "gflops": f"{gflops:.2f}" if gflops >= 0 else "N/A",
            "inference_ms": f"{avg_time * 1000:.1f}" if avg_time >= 0 else "N/A",
            "fps": f"{fps:.2f}" if fps >= 0 else "N/A",
            "peak_memory_MB": f"{peak_mem:.0f}" if peak_mem >= 0 else "N/A",
        })

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Write CSV
    csv_path = "results/efficiency/efficiency_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print(f"\n{'Model':<30} {'Params(M)':<12} {'Train(M)':<12} {'GFLOPs':<10} {'ms':<8} {'FPS':<8} {'Mem(MB)':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<30} {r['total_params_M']:<12} {r['trainable_params_M']:<12} {r['gflops']:<10} {r['inference_ms']:<8} {r['fps']:<8} {r['peak_memory_MB']:<10}")


if __name__ == "__main__":
    main()
