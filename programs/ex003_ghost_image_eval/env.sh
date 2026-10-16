#!/usr/bin/env bash
# ex003_ghost_image_eval のスクリプトを実行するための環境変数を設定する．
# 使い方: source programs/ex003_ghost_image_eval/env.sh
#
# 既存のconda環境 identity-anonymizer (environment.yml) をそのまま用いる．
# WSL2上では，PyTorch同梱のcuDNNが libcuda.so を見つけられるよう /usr/lib/wsl/lib を
# 明示的にライブラリパスへ加える必要がある．

CONDA_ENV_DIR="${CONDA_ENV_DIR:-${HOME}/miniconda3/envs/identity-anonymizer}"
export PYTHON="${CONDA_ENV_DIR}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib/python3.9/site-packages/torch/lib:/usr/lib/wsl/lib"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONDONTWRITEBYTECODE=1
