"""
実験2・実験3で共通して用いる，サンプリング・デコード・比較基準・要約統計量の関数をまとめたモジュール．
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import common


def make_generator(seed: int) -> torch.Generator:
    """
    GPU上の乱数生成器を，seedを固定して作成する．

    引数:
        seed (int): 乱数シード．
    戻り値:
        generator (torch.Generator): seedを設定済みのGPU上の乱数生成器．
    """
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def sample_standard_normal(num_images: int, num_samples: int, latent_dim: int, generator: torch.Generator) -> torch.Tensor:
    """
    標準正規分布から潜在空間のノイズをサンプリングする．

    引数:
        num_images (int): 画像数 N．
        num_samples (int): 1枚あたりのサンプル数 K．
        latent_dim (int): 潜在次元数 d．
        generator (torch.Generator): `make_generator` で作成した乱数生成器．
    戻り値:
        z (torch.Tensor): 形状 (N, K, d) の標準正規乱数 (GPU上，float32)．
    """
    return torch.randn(num_images, num_samples, latent_dim, generator=generator, device="cuda")


@torch.no_grad()
def decode_displaced(model: torch.nn.Module, mu: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """
    潜在変数の平均 mu から displacement だけ移動した点をデコードする．

    引数:
        model (torch.nn.Module): 学習済みのVAE．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        displacement (torch.Tensor): 形状 (N, K, d) の潜在空間での移動ベクトル．
    戻り値:
        v_tilde (torch.Tensor): 形状 (N, K, 512) のデコード結果．
    """
    num_images, num_samples, latent_dim = displacement.shape
    f = mu.unsqueeze(1) + displacement
    return model.decode(f.reshape(-1, latent_dim)).reshape(num_images, num_samples, -1)


def cosine_to_reference(v_tilde: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    サンプルごとのデコード結果と，画像ごとの基準ベクトルとのコサイン類似度を計算する．

    引数:
        v_tilde (torch.Tensor): 形状 (N, K, 512) のデコード結果．
        reference (torch.Tensor): 形状 (N, 512) の基準ベクトル．
    戻り値:
        cos (torch.Tensor): 形状 (N, K) のコサイン類似度．
    """
    return F.cosine_similarity(v_tilde, reference.unsqueeze(1), dim=2)


@torch.no_grad()
def nearest_other_distance(x: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """
    各行について，自分自身を除く最も近い行までのユークリッド距離を求める．

    引数:
        x (torch.Tensor): 形状 (N, D) のベクトル群．
        chunk_size (int): 1回に距離行列を計算する行数．
    戻り値:
        distance (torch.Tensor): 形状 (N,) の最近傍距離．
    """
    results = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start:start + chunk_size]
        dist = torch.cdist(chunk, x)
        rows = torch.arange(chunk.shape[0], device=x.device)
        dist[rows, start + rows] = float("inf")
        results.append(dist.min(dim=1).values)
    return torch.cat(results)


@torch.no_grad()
def nearest_other_cosine(x: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """
    各行について，自分自身を除く他の行とのコサイン類似度の最大値を求める．

    引数:
        x (torch.Tensor): 形状 (N, D) のベクトル群．
        chunk_size (int): 1回に類似度行列を計算する行数．
    戻り値:
        cos (torch.Tensor): 形状 (N,) の最近傍のコサイン類似度．
    """
    normalized = F.normalize(x, dim=1)
    results = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = normalized[start:start + chunk_size]
        sim = chunk @ normalized.T
        rows = torch.arange(chunk.shape[0], device=x.device)
        sim[rows, start + rows] = -float("inf")
        results.append(sim.max(dim=1).values)
    return torch.cat(results)


def random_other_index(num_images: int, seed: int) -> torch.Tensor:
    """
    各画像に対して，自分自身とは必ず異なる画像の番号をランダムに1つ割り当てる．
    ランダムな巡回置換を用いるため，不動点は存在しない．

    引数:
        num_images (int): 画像数 N．
        seed (int): 乱数シード．
    戻り値:
        index (torch.Tensor): 形状 (N,) の画像番号 (GPU上，int64)．index[i] != i を満たす．
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_images)
    index = np.empty(num_images, dtype=np.int64)
    index[order] = np.roll(order, -1)
    return torch.from_numpy(index).cuda()


def quantile_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    配列の平均・標準偏差・分位点を，列名の接頭辞を付けた辞書として返す．

    引数:
        values (np.ndarray): 任意の形状の数値配列．
        prefix (str): 列名の接頭辞 (例: "cos_v")．
    戻り値:
        summary (dict): `{prefix}_mean`，`{prefix}_std`，`{prefix}_q05` 等をkeyとする辞書．
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    # 重複画像に対する比のように定義できない値 (NaN) は集計から除外する
    flat = flat[~np.isnan(flat)]
    summary = {f"{prefix}_mean": float(flat.mean()), f"{prefix}_std": float(flat.std())}
    for q, value in zip(common.QUANTILES, np.quantile(flat, common.QUANTILES)):
        summary[f"{prefix}_q{int(round(q * 100)):02d}"] = float(value)
    return summary


def load_inputs() -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    実験2・実験3に共通する入力 (VAE，顔ベクトル，潜在変数の平均・標準偏差) を準備する．

    引数:
        なし．
    戻り値:
        model (torch.nn.Module): 学習済みのVAE．
        v (torch.Tensor): 形状 (N, 512) のArcFaceの顔ベクトル (GPU上)．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        is_test (np.ndarray): 形状 (N,) のbool配列．VAEの学習時に検証用だった画像が True．
    """
    from data import load_arcface_cache
    from model import encode_all, load_model

    embeddings, is_test, _ = load_arcface_cache()
    model = load_model()
    mu, std = encode_all(model, embeddings)
    return model, torch.from_numpy(embeddings).cuda(), mu, std, is_test
