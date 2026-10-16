"""
ex003_ghost_image_eval の全スクリプトで共有する設定・パス・小さな補助関数をまとめたモジュール．

既存の `src/`，`third_party/` は変更せず，`sys.path` へ追加して読み込むだけにする．
サンプリング方式，VAEの読み込み，ラベル・重複グループ・データ分割，線形の属性分類器は
`programs/ex002_partial_prior_sampling/` の実装をそのまま読み込んで再利用する
(ex002 の設定モジュール `common` と名前が衝突しないよう，本モジュールは `settings` とする)．
"""

import json
import os
import sys
import types

# cuBLASを決定論的に動作させるための設定(CUDAの初期化より前に設定する必要がある)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EXPERIMENT_NAME = "ex003_ghost_image_eval"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DEV_KIT_SRC_DIR = os.path.join(REPO_ROOT, ".ai", "ai-dev-kit", "src")
EX002_DIR = os.path.join(REPO_ROOT, "programs", "ex002_partial_prior_sampling")

for _path in (EX002_DIR, SRC_DIR, DEV_KIT_SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# mxnet は，生成した顔を元画像へ貼り戻す処理 (ランドマーク検出器 Handler) でしか使われず，dill は
# 超解像モデルの補助関数でしか使われない．本実験は貼り戻しも超解像も行わないため，これらが
# 読み込めない環境では空のモジュールで代用し，既存のモジュールを変更せずに import できるようにする
# (このプロセス内に限る．既存のconda環境やsite-packagesは変更しない)．
# torch.utils.data は dill の有無を import 時に調べるため，代用のモジュールを登録する前に読み込んでおく．
import torch.utils.data  # noqa: E402,F401

for _name in ("mxnet", "dill"):
    try:
        __import__(_name)
    except Exception:
        sys.modules[_name] = types.ModuleType(_name)

ARCFACE_CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "ex001_vae_latent_diagnosis", "arcface_utkface")
EX002_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "ex002_partial_prior_sampling")
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", EXPERIMENT_NAME)
CROP_DIR = os.path.join(DATASET_DIR, "eval_crops")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", EXPERIMENT_NAME)

# --- 実験条件 (order_003) ---
#: 乱数シード
SEED = 0
#: 本実験で用いる画像数
NUM_IMAGES = 1000
#: 試行に用いる画像数
NUM_TRIAL_IMAGES = 50
#: 予稿の再現に用いる画像数と alpha
NUM_REPRODUCTION_IMAGES = 200
REPRODUCTION_ALPHAS = [0.0, 1.0, 1.5, 2.0]
#: 予稿・発表資料に記載された値 (alpha: (cos類似度, 性別一致率, 年齢RMSE))
PAPER_VALUES = {
    0.0: (0.741, 0.929, 6.73),
    1.0: (0.6851, 0.9206, 7.1740),
    1.5: (0.6306, 0.9106, 7.5622),
    2.0: (0.5734, 0.9041, 7.8757),
}
#: 予稿の再現で許容する cos 類似度のずれ
REPRODUCTION_TOLERANCE = 0.03
#: 再識別率 top-1 の目標値
TARGET_REID = [0.9, 0.5, 0.1]
#: パラメータの較正に用いるベクトル空間でのサンプル数
CALIBRATION_NUM_SAMPLES = 10
#: 生成と評価のバッチサイズ
BATCH_SIZE = 50


def condition_dir(method: str, hyperparameter: str, seed: int) -> str:
    """
    `outputs/{実験名}/{手法}/{ハイパーパラメータ}/{seed}/` 形式の保存先ディレクトリを作成して返す．

    引数:
        method (str): 実験の種類 (例: "main")．
        hyperparameter (str): ハイパーパラメータを表す文字列 (例: "n1000")．
        seed (int): 乱数シード．
    戻り値:
        target_dir (str): 作成済みの保存先ディレクトリのパス．
    """
    target_dir = os.path.join(OUTPUT_DIR, method, hyperparameter, str(seed))
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def save_json(obj: dict, path: str) -> None:
    """
    辞書をJSONファイルとして保存する．

    引数:
        obj (dict): 保存する辞書 (JSONへ変換可能な値のみを含むこと)．
        path (str): 保存先のパス．
    戻り値:
        なし．
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """
    JSONファイルを辞書として読み込む．

    引数:
        path (str): 読み込むJSONファイルのパス．
    戻り値:
        obj (dict): 読み込んだ辞書．
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finished(target_dir: str) -> bool:
    """
    その条件の処理が正常終了済みかどうかを，完了マーカー `meta.json` の有無で判定する．

    引数:
        target_dir (str): 条件ごとの保存先ディレクトリ．
    戻り値:
        finished (bool): 正常終了済みであれば True．
    """
    return os.path.exists(os.path.join(target_dir, "meta.json"))
