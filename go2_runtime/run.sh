#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd "$(dirname "$0")" && pwd)"
export MPLBACKEND=Agg
export PYTHONPATH="$runtime_dir/.deps${PYTHONPATH:+:$PYTHONPATH}"
exec "${HIMLOCO_REAL_PYTHON:-/home/lenovo/miniconda3/envs/isclab/bin/python}" "$runtime_dir/run_ours.py" "$@"
