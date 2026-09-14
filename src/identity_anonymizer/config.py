"""
リポジトリ全体で共有するパス設定と，GHOST(third_party/ghost)を import 可能にするための
ブートストラップ処理をまとめたモジュール．

GHOST側のコード(`network`, `arcface_model`, `insightface_func`, `coordinate_reg`, `models`,
`utils` 等)は，リポジトリのルートから実行されることを前提とした絶対importで書かれている．
これを変更せずに利用するため，本モジュールを最初にimportした時点で
`third_party/ghost` を `sys.path` の先頭に追加する．
"""

import os
import sys

# リポジトリのルートディレクトリ(このファイルから2階層上: src/identity_anonymizer/config.py -> repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GHOST_ROOT = os.path.join(REPO_ROOT, "third_party", "ghost")

WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
GHOST_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "ghost")
DEX_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "dex")
ANONYMIZER_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "anonymizers")

DATA_DIR = os.path.join(REPO_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


def ensure_ghost_importable():
    """
    GHOST(third_party/ghost)配下のモジュールをimportできるよう，`sys.path` へ
    そのルートディレクトリを追加する．

    引数:
        なし．
    戻り値:
        なし．
    """
    if not os.path.isdir(GHOST_ROOT):
        raise FileNotFoundError(
            f"GHOSTのsubmoduleが見つかりません: {GHOST_ROOT}\n"
            "`git submodule update --init` を実行してください．"
        )
    if GHOST_ROOT not in sys.path:
        sys.path.insert(0, GHOST_ROOT)


ensure_ghost_importable()
