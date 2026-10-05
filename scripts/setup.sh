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
GHOST_DIR="third_party/ghost"
GHOST_URL="$(git config -f .gitmodules "submodule.${GHOST_DIR}.url")"
# submoduleが指すコミット．.git が無い場合(zip展開や rm -rf .git 後)は参照先を
# gitから取得できないためここに固定する．submoduleを更新した場合はここも更新すること．
GHOST_COMMIT="44e58aad8600ee83ad4c8213aaa9698ccaf66c1c"

if git ls-files --stage -- "${GHOST_DIR}" 2>/dev/null | grep -q '^160000 '; then
    git submodule update --init "${GHOST_DIR}"
    if ! git ls-files --stage -- "${GHOST_DIR}" | grep -q "^160000 ${GHOST_COMMIT} "; then
        echo "警告: submoduleの参照先が scripts/setup.sh の GHOST_COMMIT と一致しません．GHOST_COMMIT を更新してください．"
    fi
else
    # 本リポジトリがGit管理下に無い，またはsubmoduleが未登録の場合は，
    # GHOSTを単独のリポジトリとして同じコミットで取得する
    echo "submoduleとして登録されていないため，${GHOST_URL} から直接取得します．"
    mkdir -p "${GHOST_DIR}"
    # 削除済みの親 .git/modules を指したままのgitfile(GHOST内のsubmoduleのものを含む)が
    # 残っているとgitコマンドが失敗するため取り除く
    find "${GHOST_DIR}" -name .git -type f | while read -r gitfile; do
        if ! git -C "$(dirname "${gitfile}")" rev-parse --git-dir >/dev/null 2>&1; then
            rm -f "${gitfile}"
        fi
    done
    if [ ! -d "${GHOST_DIR}/.git" ]; then
        git -C "${GHOST_DIR}" init -q
    fi
    if [ "$(git -C "${GHOST_DIR}" rev-parse -q --verify HEAD 2>/dev/null || true)" != "${GHOST_COMMIT}" ]; then
        git -C "${GHOST_DIR}" fetch -q --depth 1 "${GHOST_URL}" "${GHOST_COMMIT}"
        git -C "${GHOST_DIR}" checkout -q -f --detach FETCH_HEAD
    fi
fi

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
