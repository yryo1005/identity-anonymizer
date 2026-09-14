#!/usr/bin/env bash
# GHOST(third_party/ghost)およびArcFace/insightface由来の学習済み重みを取得し，
# weights/ghost/ 以下へ配置する．
#
# 取得元URLは third_party/ghost/download_models.sh (ai-forever/ghost 由来) と同一である．
# arcface_model/iresnet.py はGit管理されておらずリリースからダウンロードする形になっているため，
# 重みと合わせてソースファイルもここで取得する．

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GHOST_ROOT="${REPO_ROOT}/third_party/ghost"
DST="${REPO_ROOT}/weights/ghost"

mkdir -p "${DST}/antelope"

echo "== ArcFace =="
wget -nc -P "${GHOST_ROOT}/arcface_model" https://github.com/sberbank-ai/sber-swap/releases/download/arcface/iresnet.py
wget -nc -P "${DST}" https://github.com/sberbank-ai/sber-swap/releases/download/arcface/backbone.pth

echo "== insightfaceの顔検出・ランドマーク(antelope) =="
wget -nc -P "${DST}/antelope" https://github.com/sberbank-ai/sber-swap/releases/download/antelope/glintr100.onnx
wget -nc -P "${DST}/antelope" https://github.com/sberbank-ai/sber-swap/releases/download/antelope/scrfd_10g_bnkps.onnx

echo "== 顔ランドマーク検出(2d106det) =="
# 2d106det-*.{params,json} は third_party/ghost (submodule) 側に直接コミットされているため，
# ダウンロードではなくsubmodule内からコピーする
cp -n "${GHOST_ROOT}/coordinate_reg/model/2d106det-0000.params" "${DST}/"
cp -n "${GHOST_ROOT}/coordinate_reg/model/2d106det-symbol.json" "${DST}/"

echo "== GHOST生成器(G_unet_2blocks) =="
wget -nc -P "${DST}" https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_2blocks.pth

echo "== 超解像モデル(任意，use_sr=Trueの場合のみ必要) =="
wget -nc -P "${DST}" https://github.com/sberbank-ai/sber-swap/releases/download/super-res/10_net_G.pth

echo "完了: ${DST} に重みを配置しました．"
