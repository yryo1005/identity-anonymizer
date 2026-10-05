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

# conda環境を有効化し忘れると，重みのダウンロード後に pip install が失敗するため，先に検査する
if ! python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 9))' 2>/dev/null; then
    echo "エラー: Python 3.9 の環境で実行してください(現在: $(python --version 2>&1))．" >&2
    echo "  conda env create -f environment.yml  # 未作成の場合" >&2
    echo "  conda activate identity-anonymizer" >&2
    exit 1
fi

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
