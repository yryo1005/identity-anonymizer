#!/usr/bin/env bash
# セットアップを一括で行うスクリプト．
# 1. GHOST(third_party/ghost) submoduleの初期化
# 2. GHOST側への最小限のパッチ適用
# 3. 重みのダウンロード(GHOST由来 + 本リポジトリのGitHub Release由来)
#
# 実行前に conda 環境(environment.yml)を作成・有効化しておくこと．

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "== 1. GHOST submoduleの初期化 =="
git submodule update --init third_party/ghost

echo "== 2. GHOSTへの最小限のパッチ適用 =="
PATCH_FILE="${REPO_ROOT}/patches/0001-silence-inner-progress-bars.patch"
if git -C third_party/ghost apply --check "${PATCH_FILE}" 2>/dev/null; then
    git -C third_party/ghost apply "${PATCH_FILE}"
    echo "patches/0001-silence-inner-progress-bars.patch を適用しました．"
else
    echo "パッチは適用済み，または対象箇所が変更されています(スキップ)．"
fi

echo "== 3. 重みのダウンロード =="
bash "${REPO_ROOT}/scripts/download_ghost_weights.sh"
bash "${REPO_ROOT}/scripts/download_release_weights.sh" || \
    echo "警告: 独自重み(DEX/VAE)の取得に失敗しました．README.md の手順に従いGitHub Releaseの設定を確認してください．"

echo "== 4. パッケージのインストール =="
pip install -e "${REPO_ROOT}"

echo "セットアップが完了しました．"
