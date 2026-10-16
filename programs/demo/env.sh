#!/usr/bin/env bash
# 匿名化デモ (programs/demo) を実行するための環境変数を設定する．
# 使い方: source programs/demo/env.sh
#
# 貼り戻しに mxnet が必要なため，ex003 で作成した環境 identity-anonymizer-mxcpu を用いる．
# この環境の作り方は programs/ex003_ghost_image_eval/env_mxcpu.sh を，デモ用に追加したライブラリは
# programs/demo/requirements_demo.txt を参照すること．

CONDA_ENV_DIR="${CONDA_ENV_DIR:-${HOME}/miniconda3/envs/identity-anonymizer-mxcpu}"
export PYTHON="${CONDA_ENV_DIR}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib/python3.9/site-packages/torch/lib:/usr/lib/wsl/lib"
export PYTHONDONTWRITEBYTECODE=1
