#!/usr/bin/env bash
# 本リポジトリ独自の重み(DEXの年齢/性別推定モデル，学習済みAnonymizer)を，
# 本リポジトリのGitHub Releaseから取得する．
#
# これらの重みはGHOST/ArcFace由来のものと異なり再配布元が無いため，本リポジトリの
# GitHub Release (https://github.com/yryo1005/identity-anonymizer/releases/tag/v1.0.0)
# にアップロード済みのものを取得する．別のRelease先を使う場合は，環境変数
# RELEASE_OWNER/RELEASE_REPO/RELEASE_TAG で上書きすること．

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OWNER="${RELEASE_OWNER:-yryo1005}"
REPO="${RELEASE_REPO:-identity-anonymizer}"
TAG="${RELEASE_TAG:-v1.0.0}"
BASE_URL="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}"

mkdir -p "${REPO_ROOT}/weights/dex" "${REPO_ROOT}/weights/anonymizers/vae"

echo "== DEXの年齢/性別推定モデル =="
wget -nc -P "${REPO_ROOT}/weights/dex" "${BASE_URL}/age_sd.pth"
wget -nc -P "${REPO_ROOT}/weights/dex" "${BASE_URL}/gender_sd.pth"

echo "== 学習済みVAE (Anonymizer) =="
wget -nc -P "${REPO_ROOT}/weights/anonymizers/vae" "${BASE_URL}/vae_512_128.pt"

echo "完了: ${REPO_ROOT}/weights に重みを配置しました．"
echo "(取得元: https://github.com/${OWNER}/${REPO}/releases/tag/${TAG})"
