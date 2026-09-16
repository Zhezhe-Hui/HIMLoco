#!/bin/bash
# ============================================================
# HIMLoco 环境一键安装脚本
# 实测环境: Ubuntu 22.04 / Python 3.8.20 / torch 1.13.1+cu117
#           RTX 4060 (sm_89) / Isaac Gym Preview 4 (仓库自带 libraries/isaacgym)
# 用法: bash setup_himloco_env.sh
# ============================================================
set -e

ENV_NAME=himloco
CONDA_BIN=/home/fabu/miniconda3/bin/conda
REPO=/home/fabu/HIMLoco
# 镜像: 清华 anaconda(conda可用) + ustc pypi(pip可用)
# 注意: 本机 openvpn 会劫持默认路由, conda官方源与 download.pytorch.org 均不通,
#       必须使用以下国内镜像
CONDA_MAIN=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
CONDA_FREE=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
PIP_IDX=https://pypi.mirrors.ustc.edu.cn/simple/

# 关键: 必须 unset PYTHONPATH, 否则 ROS Humble 的 python3.10 包会污染本环境
export PYTHONPATH=

echo "[1/5] 创建 conda 环境 ($ENV_NAME, Python 3.8)"
$CONDA_BIN create -y -n $ENV_NAME python=3.8 \
    --override-channels -c $CONDA_MAIN -c $CONDA_FREE

ENV=/home/fabu/miniconda3/envs/$ENV_NAME
PIP="$ENV/bin/python -m pip"
# torch 的 cpp_extension 通过 PATH 查找 ninja, 必须把 env/bin 放进 PATH
export PATH=$ENV/bin:$PATH
# gym_38.so 依赖 libpython3.8.so.1.0, 位于 env/lib
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

echo "[2/5] 安装 pip 基础工具"
$PIP install --upgrade "pip<25.1" setuptools==59.5.0 wheel -i $PIP_IDX

echo "[3/5] 安装 PyTorch 1.13.1"
# torch 1.13.1 wheel 文件名必须符合规范, 否则 pip 报 "not a valid wheel filename"
# 若 /tmp 无缓存, 取消下面注释让 pip 直接下载:
$PIP install torch==1.13.1 -i $PIP_IDX
$PIP install torchvision==0.14.1 -i $PIP_IDX

echo "[4/5] 安装依赖"
# numpy 必须 <1.24 (Isaac Gym 依赖已废弃的 np.float 别名)
# ninja 必需 (gymtorch 需 JIT 编译 C++ 扩展)
$PIP install numpy==1.21.6 ninja==1.11.1.1 -i $PIP_IDX
$PIP install scipy pyyaml pillow matplotlib tensorboard==2.11.2 \
           scikit-image trimesh -i $PIP_IDX
# scikit-fmm: PyPI 无 Linux wheel, 需 gcc 源码编译 (FMM 规划器依赖)
$PIP install scikit-fmm -i $PIP_IDX

echo "[5/5] 安装本地三个包 (必须 --no-deps)"
# --no-deps 至关重要:
#   - legged_gym 声明依赖 'isaacgym', PyPI 上不存在
#   - legged_gym 声明依赖 'rsl-rl', PyPI 上是另一个项目, 会覆盖本仓库改过的版本
$PIP install --no-deps -e $REPO/libraries/isaacgym/python
$PIP install --no-deps -e $REPO/rsl_rl
$PIP install --no-deps -e $REPO/legged_gym

echo
echo "===== 安装完成, 验证 ====="
$ENV/bin/python - <<'PY'
import torch, numpy as np, isaacgym, skfmm, skimage, scipy, trimesh
print("torch      :", torch.__version__, "| CUDA可用:", torch.cuda.is_available())
print("GPU        :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("numpy      :", np.__version__)
print("isaacgym   : OK")
print("skfmm      : OK")
a=torch.randn(512,512,device='cuda'); b=torch.randn(512,512,device='cuda')
torch.cuda.synchronize(); print("GPU matmul : OK  (sm_89 经 PTX JIT 可用)")
PY

cat <<'TIP'

===== 启动训练 =====
cd /home/fabu/HIMLoco/legged_gym/legged_gym/scripts
conda activate himloco
# 必须 unset PYTHONPATH, 否则被 ROS 包污染
export PYTHONPATH=
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 显存关键: go2_config 中 stone.num_stones=1000 是"每个env"的石头数,
# 8GB 显存下 num_envs 上限约 16 (16*1005 actor)。超过会 PhysX OOM。
python train.py --task=go2 --headless --num_envs=16 --max_iterations=1500

# 纯底层步态训练(不需石头)时可临时置 stone.num_stones=0, num_envs 可达 256+
TIP
