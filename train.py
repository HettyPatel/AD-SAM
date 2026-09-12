#!/usr/bin/env python
"""Single training entry point.

    python train.py adsam     [options]   # AD-SAM (frozen SAM ViT-H + ResNet-50), see --help
    python train.py deeplabv3 [options]   # DeepLabV3-ResNet101 baseline

Both share the dataset, validation protocol, metric and output layout (results/).
"""
import sys

MODELS = {
    'adsam': 'adsam.training.train_adsam',
    'deeplabv3': 'adsam.training.train_deeplabv3',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print(__doc__)
        sys.exit(2)
    import importlib
    importlib.import_module(MODELS[sys.argv[1]]).main(sys.argv[2:])


if __name__ == '__main__':
    main()
