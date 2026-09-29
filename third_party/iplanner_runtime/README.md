# iPlanner runtime files

This directory contains the minimal official iPlanner network and trajectory
interpolation modules needed to load the pretrained checkpoint in HIMLoco.

- Upstream: https://github.com/leggedrobotics/iPlanner
- Upstream commit: `4a8d823ff9d09c3f626b727e7e00484b38f80d49`
- License: MIT; see `LICENSE`
- Checkpoint: `assets/checkpoints/iplanner/plannernet.pt`
- Checkpoint SHA-256: `0f61736007c8a7b28d463eb359450369bd0ad4ef21104148c166837d2a01e3b4`

The official network and cubic interpolation are unchanged. The HIMLoco
adapter in `legged_gym/legged_gym/envs/go2/iplanner_controller.py` handles
metric depth resizing, local-goal clipping, path following, and conversion to
the shared `vx/wz` command interface.
