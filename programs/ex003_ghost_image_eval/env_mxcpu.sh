#!/usr/bin/env bash
# 貼り戻した画像での評価 (pasteback.py) を実行するための環境変数を設定する．
# 使い方: source programs/ex003_ghost_image_eval/env_mxcpu.sh
#
# 既存のconda環境 identity-anonymizer を複製した identity-anonymizer-mxcpu を用いる．
# 複製後に行った変更は次の3点のみである (既存の環境は変更していない)．
#   pip uninstall -y mxnet-cu112
#   pip install --no-deps mxnet==1.9.1 dill        # CPU版の mxnet
#   pip install --no-deps numpy==1.23.5            # mxnet 1.9.1 は numpy>=1.24 で import できないため

CONDA_ENV_DIR="${CONDA_ENV_DIR:-${HOME}/miniconda3/envs/identity-anonymizer-mxcpu}"
export PYTHON="${CONDA_ENV_DIR}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib/python3.9/site-packages/torch/lib:/usr/lib/wsl/lib"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONDONTWRITEBYTECODE=1
