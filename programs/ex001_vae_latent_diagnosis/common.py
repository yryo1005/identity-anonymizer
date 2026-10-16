"""
ex001_vae_latent_diagnosis の全スクリプトで共有する設定・パス・小さな補助関数をまとめたモジュール．

既存の `src/`，`third_party/` は変更せず，`sys.path` へ追加して読み込むだけにする．
"""

import json
import os
import sys

# cuBLASを決定論的に動作させるための設定(CUDAの初期化より前に設定する必要がある)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EXPERIMENT_NAME = "ex001_vae_latent_diagnosis"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DEV_KIT_SRC_DIR = os.path.join(REPO_ROOT, ".ai", "ai-dev-kit", "src")

for _path in (SRC_DIR, DEV_KIT_SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- 入力 ---
UTKFACE_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "UTKFace", "images")
VAE_WEIGHT_PATH = os.path.join(REPO_ROOT, "weights", "anonymizers", "vae", "vae_512_128.pt")
REFERENCE_TEST_EMBEDS_PATH = os.path.join(REPO_ROOT, "data", "processed", "test_source_embeds.npy")

# --- 出力 ---
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", EXPERIMENT_NAME, "arcface_utkface")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", EXPERIMENT_NAME)

# --- 実験条件 (order_001) ---
#: サンプリング時に標準偏差へ乗じるスケール係数 alpha (既存実装の noise_level に相当)
ALPHAS = [0.0, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
#: 1枚の画像・1つの alpha あたりのサンプル数 K
NUM_SAMPLES = 10
#: 乱数シード
SEEDS = [0, 1, 2]
#: 実験3で用いる半径の倍率 (半径 = 倍率 * sqrt(d))
RADIUS_FACTORS = [0.5, 1.0, 1.5, 2.0]
#: Active Units の判定閾値 (Burda et al. 2016)
ACTIVE_UNIT_THRESHOLD = 0.01
#: 要約統計量として出力する分位点
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def condition_dir(method: str, hyperparameter: str, seed: int) -> str:
    """
    `outputs/{実験名}/{手法}/{ハイパーパラメータ}/{seed}/` 形式の保存先ディレクトリを作成して返す．

    引数:
        method (str): 実験の種類 (例: "ex2_displacement")．
        hyperparameter (str): ハイパーパラメータを表す文字列 (例: "K10")．
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
    その条件の実験が正常終了済みかどうかを，完了マーカー `meta.json` の有無で判定する．

    引数:
        target_dir (str): 条件ごとの保存先ディレクトリ．
    戻り値:
        finished (bool): 正常終了済みであれば True．
    """
    return os.path.exists(os.path.join(target_dir, "meta.json"))
