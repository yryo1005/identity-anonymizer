#!/usr/bin/env bash
# 本リポジトリ独自の重み(DEXの年齢/性別推定モデル，学習済みAnonymizer)を，
# 本リポジトリのGitHub Releaseから取得する．
#
# これらの重みはGHOST/ArcFace由来のものと異なり再配布元が無いため，本リポジトリの
# GitHub Releaseにアップロードしたうえで取得する運用とする．
# <OWNER>/<REPO>/<TAG> は，実際にReleaseを作成した後に書き換えること．

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OWNER="${RELEASE_OWNER:-<OWNER>}"
REPO="${RELEASE_REPO:-identity-anonymizer}"
TAG="${RELEASE_TAG:-weights-v1}"
BASE_URL="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}"

mkdir -p "${REPO_ROOT}/weights/dex" "${REPO_ROOT}/weights/anonymizers/vae"

echo "== DEXの年齢/性別推定モデル =="
wget -nc -P "${REPO_ROOT}/weights/dex" "${BASE_URL}/age_sd.pth"
wget -nc -P "${REPO_ROOT}/weights/dex" "${BASE_URL}/gender_sd.pth"

echo "== 学習済みVAE (Anonymizer) =="
wget -nc -P "${REPO_ROOT}/weights/anonymizers/vae" "${BASE_URL}/vae_512_128.pt"

echo "完了: ${REPO_ROOT}/weights に重みを配置しました．"
echo "OWNER/REPO/TAGが未設定の場合は環境変数 RELEASE_OWNER/RELEASE_REPO/RELEASE_TAG で指定してください．"
