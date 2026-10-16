"""
学習済みの `VAEAnonymizer` を読み込み，診断に必要な量 (mu，標準偏差，デコード結果) を計算するモジュール．

モデルの定義は既存の `identity_anonymizer.anonymizers.vae.VAEAnonymizer` をそのまま用いる．
既存実装のエンコーダは対数分散 logvar を出力し，標準偏差を std = exp(0.5 * logvar) として
サンプリングする．本モジュールではこの std を一貫して用いる．
"""

from typing import Tuple

import numpy as np
import torch

import common
from machine_learning_utils import set_seed
from identity_anonymizer.anonymizers.vae import VAEAnonymizer


def load_model(ModelClass=VAEAnonymizer, weight_path: str = None, seed: int = 0) -> torch.nn.Module:
    """
    モデルをインスタンス化し，学習済みの重みを読み込む．潜在次元数は重みの形状から決定する．

    引数:
        ModelClass (type): `torch.nn.Module` のクラス (既定は `VAEAnonymizer`)．
        weight_path (str): 学習済み重みのパス．None の場合は `common.VAE_WEIGHT_PATH`．
        seed (int): 乱数シード．
    戻り値:
        model (torch.nn.Module): GPU上に配置した評価モードのモデル．
    """
    set_seed(seed)
    weight_path = weight_path or common.VAE_WEIGHT_PATH
    state_dict = torch.load(weight_path, map_location="cpu")
    latent_dim = state_dict["fc_mu.weight"].shape[0]

    model = ModelClass(latent_dim=latent_dim, seed=seed)
    model.load_state_dict(state_dict)
    return model.cuda().eval()


@torch.no_grad()
def encode_all(model: torch.nn.Module, embeddings: np.ndarray, batch_size: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    全ての顔ベクトルをエンコードし，潜在変数の平均と標準偏差を求める．

    引数:
        model (torch.nn.Module): `load_model` で読み込んだVAE．
        embeddings (np.ndarray): 形状 (N, 512) の顔ベクトル．
        batch_size (int): 1回に処理する件数．
    戻り値:
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均 (GPU上，float32)．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差 exp(0.5 * logvar) (GPU上，float32)．
    """
    v = torch.from_numpy(embeddings).cuda()
    mus, stds = [], []
    for chunk in v.split(batch_size):
        mu, logvar = model.encode(chunk)
        mus.append(mu)
        stds.append(torch.exp(0.5 * logvar))
    return torch.cat(mus), torch.cat(stds)


def describe_weights(weight_path: str = None) -> dict:
    """
    学習済み重みに含まれる各テンソルの形状を取得する (潜在次元数の確認用)．

    引数:
        weight_path (str): 学習済み重みのパス．None の場合は `common.VAE_WEIGHT_PATH`．
    戻り値:
        shapes (dict): パラメータ名をkey，形状のリストをvalueとする辞書．
    """
    state_dict = torch.load(weight_path or common.VAE_WEIGHT_PATH, map_location="cpu")
    return {name: list(tensor.shape) for name, tensor in state_dict.items()}
