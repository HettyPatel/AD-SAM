import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_CRF = True
except ImportError:
    _HAS_CRF = False

def crf_postprocess(image, logits, n_classes=19, iter_max=5):
    """
    Refines the segmentation logits using DenseCRF, using parameters from the original notebook.
    
    Args:
        image (np.array): RGB image (H, W, 3) in uint8 or float [0-1].
        logits (np.array): Logits shape [C,H,W], [1,C,H,W], or [H,W,C].
        n_classes (int): Number of classes.
        iter_max (int): CRF iterations (default=5).
    Returns:
        np.array: Refined mask (H, W).
    """
    if not _HAS_CRF:
        raise RuntimeError(
            "pydensecrf is not installed. On Windows, install it with:\n"
            "  pip install git+https://github.com/lucasb-eyer/pydensecrf.git\n"
            "Or run without --apply_crf."
        )

    # 1) To uint8 [0–255], then C-contiguous
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    # 2) Force logits contiguous
    logits = np.ascontiguousarray(logits)
    # 3) Handle [H,W,C] → [C,H,W]
    if logits.ndim == 3 and logits.shape[2] == n_classes:
        logits = logits.transpose(2, 0, 1)
    # 4) Handle [1,C,H,W] → [C,H,W]
    if logits.ndim == 4:
        logits = logits.squeeze(0)

    # 5) Softmax → [C,H,W]
    probs = F.softmax(torch.from_numpy(logits), dim=0).cpu().numpy()

    # 6) Unary potentials
    flat = probs.reshape((n_classes, -1))
    unary = unary_from_softmax(flat)
    unary = np.ascontiguousarray(unary)

    # 7) Setup CRF
    H, W, _ = image.shape
    d = dcrf.DenseCRF2D(W, H, n_classes)
    d.setUnaryEnergy(unary)
    # Gaussian term (small spatial kernel)
    d.addPairwiseGaussian(
        sxy=(5, 5),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )
    # Bilateral term (small spatial + moderate color kernel)
    d.addPairwiseBilateral(
        sxy=(5, 5),
        srgb=(13, 13, 13),
        rgbim=image,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # 8) Inference
    Q = d.inference(iter_max)
    refined = np.argmax(Q, axis=0).reshape((H, W))
    return refined


if __name__ == "__main__":
    
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_logits = np.random.randn(1, 19, 256, 256).astype(np.float32)
    out = crf_postprocess(dummy_img, dummy_logits, n_classes=19, iter_max=5)
    print("Refined shape:", out.shape, "Labels:", np.unique(out))
