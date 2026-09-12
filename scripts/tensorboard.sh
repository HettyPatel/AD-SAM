#!/bin/bash
# Serve TensorBoard for all runs (results/tensorboard) on localhost:6006.
# Remote machine:  ssh -N -L 6006:localhost:6006 user@host   then open http://localhost:6006
cd "$(dirname "$0")/.."
exec tensorboard --logdir results/tensorboard --port "${PORT:-6006}" --host localhost --reload_interval 30
