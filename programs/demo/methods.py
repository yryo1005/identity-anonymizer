"""
ブラウザから選べる匿名化の方式の定義と，方式に応じたソースの顔ベクトルの生成．

方式の実装は ex002 の `samplers.py` をそのまま用いる．プリセットの値は，report_004 の較正
(ベクトル空間での再識別率 top-1 が約 0.9 / 0.5 / 0.1 になるパラメータ) に基づく．
"""

import json
from typing import Dict, List

import numpy as np
import torch

import demo_settings  # noqa: F401  (既存のモジュールを読み込むためのパスの設定)
import settings  # noqa: F401  (ex003: src/ のパスの設定と，mxnet が無い環境での代用)
import common as ex002_common
import samplers

ex002_common.NUM_SAMPLES = 1

#: ブラウザに提示する方式の一覧．params の各要素は (名前, 表示名, 最小値, 最大値, 刻み, 既定値)．
METHODS: List[dict] = [
    {
        "id": "M3u", "label": "M3u：全次元を事前分布へ寄せる（推奨）",
        "description": "全ての潜在次元で f = γμ + √(1−γ²)z とする．γ が小さいほど本人の成分が置き換わり，別人に近づく．"
                       "強い匿名化でも顔が崩れにくい．",
        "params": [{"name": "gamma", "label": "γ（小さいほど匿名性が高い）", "min": 0.0, "max": 0.95, "step": 0.01, "default": 0.42}],
        "presets": {"weak": {"gamma": 0.55}, "medium": {"gamma": 0.42}, "strong": {"gamma": 0.28}},
    },
    {
        "id": "M2", "label": "M2：属性に関連する k 次元を保持し，残りを事前分布から取る",
        "description": "属性関連度の上位 k 次元は事後分布 μ + σz から，残りの次元は N(0, 1) から取る．"
                       "k が小さいほど匿名性が高い．",
        "params": [{"name": "k", "label": "保持する次元数 k（小さいほど匿名性が高い）", "min": 0, "max": 128, "step": 1, "default": 49}],
        "presets": {"weak": {"k": 68}, "medium": {"k": 49}, "strong": {"k": 32}},
    },
    {
        "id": "M3", "label": "M3：M2 のソフト版",
        "description": "上位 k 次元は事後分布から，残りの次元は γμ + √(1−γ²)z とする．",
        "params": [
            {"name": "k", "label": "保持する次元数 k", "min": 0, "max": 128, "step": 1, "default": 32},
            {"name": "gamma", "label": "残りの次元の γ（小さいほど匿名性が高い）", "min": 0.0, "max": 0.95, "step": 0.01, "default": 0.5},
        ],
        "presets": {},
    },
    {
        "id": "M0", "label": "M0：現行方式（α スケーリング）",
        "description": "f = μ + α σ ⊙ z（予稿の方式）．α = 0 は VAE の再構成のみ．α を大きくすると匿名性が上がるが，"
                       "α ≥ 5 では顔が崩れやすい．",
        "params": [{"name": "alpha", "label": "α（大きいほど匿名性が高い）", "min": 0.0, "max": 7.0, "step": 0.1, "default": 2.0}],
        "presets": {"weak": {"alpha": 3.5}, "medium": {"alpha": 5.1}, "strong": {"alpha": 7.0}},
    },
    {
        "id": "Self", "label": "Self：匿名化しない（比較用）",
        "description": "元の顔ベクトルをそのまま使って自分自身と顔交換する．GHOST 自体による変化を確認するための比較用．",
        "params": [], "presets": {},
    },
]

METHOD_IDS = [method["id"] for method in METHODS]


def load_ranking() -> np.ndarray:
    """
    M2・M3 で保持する次元を選ぶための，属性関連度の降順に並べた潜在次元の番号を読み込む．

    引数:
        なし．
    戻り値:
        ranking (np.ndarray): 形状 (d,) の次元の番号．
    """
    with open(demo_settings.RANKING_PATH, "r", encoding="utf-8") as f:
        return np.array(json.load(f)["ranking"])


def validate(method_id: str, params: Dict[str, float]) -> Dict[str, float]:
    """
    方式とパラメータを検証し，不足しているパラメータを既定値で補う．

    引数:
        method_id (str): 方式の識別子．
        params (dict): ブラウザから送られたパラメータ．
    戻り値:
        checked (dict): 範囲内に収めたパラメータ．
    """
    if method_id not in METHOD_IDS:
        raise ValueError(f"未知の方式です: {method_id}")
    method = METHODS[METHOD_IDS.index(method_id)]
    checked = {}
    for spec in method["params"]:
        value = float(params.get(spec["name"], spec["default"]))
        value = min(max(value, spec["min"]), spec["max"])
        checked[spec["name"]] = int(round(value)) if spec["step"] == 1 else value
    return checked


def make_setting(method_id: str, params: Dict[str, float]) -> samplers.Setting:
    """
    方式とパラメータから，ex002 のサンプリング設定を作る．

    引数:
        method_id (str): 方式の識別子 ("M0"，"M2"，"M3"，"M3u")．
        params (dict): `validate` で検証済みのパラメータ．
    戻り値:
        setting (samplers.Setting): サンプリング設定．
    """
    if method_id == "M0":
        return samplers.Setting("M0", "M0", "alpha", params["alpha"], alpha=params["alpha"])
    if method_id == "M2":
        return samplers.Setting("M2", "M2", "k", params["k"], k=params["k"])
    if method_id == "M3":
        return samplers.Setting("M3", "M3", "gamma", params["gamma"], k=params["k"], gamma=params["gamma"])
    return samplers.Setting("M3u", "M3u", "gamma", params["gamma"], k=0, gamma=params["gamma"])


@torch.no_grad()
def make_sources(vae: torch.nn.Module, embeddings: torch.Tensor, method_id: str, params: Dict[str, float],
                 seed: int, fix_norm: bool, ranking: np.ndarray) -> torch.Tensor:
    """
    人物ごとの顔ベクトルから，選んだ方式で匿名化後の顔ベクトル (GHOSTのソース) を作る．

    引数:
        vae (torch.nn.Module): 学習済みのVAE．
        embeddings (torch.Tensor): 形状 (P, 512) の人物ごとの顔ベクトル (RGBの順序の顔画像から計算したもの)．
        method_id (str): 方式の識別子．
        params (dict): `validate` で検証済みのパラメータ．
        seed (int): 乱数シード．
        fix_norm (bool): True の場合，出力を実データの平均ノルムにスケーリングする．
        ranking (np.ndarray): 形状 (d,) の属性関連度の順位．
    戻り値:
        sources (torch.Tensor): 形状 (P, 512) のソースの顔ベクトル．
    """
    if method_id == "Self":
        return embeddings
    mu, logvar = vae.encode(embeddings)
    std = torch.exp(0.5 * logvar)
    v0 = vae.decode(mu)
    setting = make_setting(method_id, params)
    mask = samplers.kept_mask(setting, ranking, ranking)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    sources = samplers.sample(setting, vae, embeddings, v0, mu, std, mask, generator, None)[:, 0]
    if fix_norm:
        sources = sources * (demo_settings.REAL_MEAN_NORM / sources.norm(dim=1, keepdim=True))
    return sources
