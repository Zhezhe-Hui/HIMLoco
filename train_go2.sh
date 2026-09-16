#!/bin/bash
# 便捷启动: bash train_go2.sh [num_envs] [max_iterations]
# 注意 8GB 显存下 num_envs 上限约 16 (go2_config 每env有1000个石头)
NUM_ENVS=${1:-16}
ITERS=${2:-1500}
ENV=/home/fabu/miniconda3/envs/himloco

# 关键环境变量(三者缺一不可)
export PYTHONPATH=                                  # 防止 ROS Humble python3.10 包污染
export PATH=$ENV/bin:$PATH                          # torch cpp_extension 需从 PATH 找 ninja
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH    # gym_38.so 需 libpython3.8.so.1.0

cd "$ENV/../../HIMLoco/legged_gym/legged_gym/scripts" 2>/dev/null || cd /home/fabu/HIMLoco/legged_gym/legged_gym/scripts
exec $ENV/bin/python train.py --task=go2 --headless \
     --num_envs=$NUM_ENVS --max_iterations=$ITERS
