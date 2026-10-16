"""
実験Bで比較するサンプリング方式の定義．

潜在空間の方式 (M0，M2，M2r，M3，M3u) は全て

    f_j = a_j * mu_j + n_j * z_j,   z ~ N(0, I)

の形に書ける．ここで a_j は mu に掛ける係数，n_j はノイズの標準偏差である．「保持する次元」の
集合を S とすると，S では a_j = 1，n_j = alpha * s_j (事後分布)，S 以外では a_j = gamma，
n_j = c (定数) である．各方式は (S, alpha, gamma, c) の違いとして表現できる．

    M0 (alpha)    : S = 全次元，n_j = alpha * s_j
    M2 (k)        : S = 属性関連度の上位 k 次元，S 以外は gamma = 0，c = 1 (事前分布)
    M2r (k)       : S = ランダムな k 次元，それ以外は M2 と同じ
    M3 (k, gamma) : S = 上位 k 次元，S 以外は gamma，c = sqrt(1 - gamma^2)
    M3u (gamma)   : S = 空集合，全次元で gamma，c = sqrt(1 - gamma^2)

出力空間の方式 (M1，M1b) は，基準ベクトルに等方的なノイズを直接足す．
"""

import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

import common


@dataclass
class Setting:
    """
    1つのサンプリング設定．

    属性:
        method (str): 方式の記号 ("M0"，"M1"，"M1b"，"M2"，"M2r"，"M3"，"M3u"，"REF")．
        curve (str): トレードオフ曲線の名前 (同じ曲線に属する設定を1本の線で結ぶ)．
        param_name (str): 掃引するパラメータの名前．
        param (float): 掃引するパラメータの値．
        k (int または None): 保持する次元数 (該当しない方式では None)．
        alpha (float): 保持する次元の標準偏差に乗じる係数．
        gamma (float): 保持しない次元の mu に乗じる係数．
        tau (float): 出力空間の方式のノイズの大きさ．
    """

    method: str
    curve: str
    param_name: str
    param: float
    k: Optional[int] = None
    alpha: float = 1.0
    gamma: float = 0.0
    tau: float = 0.0

    @property
    def name(self) -> str:
        """設定を一意に識別する名前を返す (戻り値: str)．"""
        return f"{self.curve}|{self.param_name}={self.param:g}"

    @property
    def is_latent(self) -> bool:
        """潜在空間でサンプリングする方式かどうかを返す (戻り値: bool)．"""
        return self.method in ("M0", "M2", "M2r", "M3", "M3u")


def all_settings() -> List[Setting]:
    """
    実験Bで評価する全ての設定を列挙する．

    引数:
        なし．
    戻り値:
        settings (list[Setting]): 設定のリスト．
    """
    settings = [Setting("M0", "M0", "alpha", a, alpha=a) for a in common.M0_ALPHAS]
    settings += [Setting("M1", "M1", "tau", t, tau=t) for t in common.M1_TAUS]
    settings += [Setting("M1b", "M1b", "tau", t, tau=t) for t in common.M1_TAUS]
    settings += [Setting("M2", "M2", "k", k, k=k) for k in common.M2_KS]
    settings += [Setting("M2r", "M2r", "k", k, k=k) for k in common.M2_KS]
    settings += [
        Setting("M3", f"M3_k{k}", "gamma", g, k=k, gamma=g)
        for k, g in itertools.product(common.M3_KS, common.M3_GAMMAS)
    ]
    settings += [Setting("M3u", "M3u", "gamma", g, k=0, gamma=g) for g in common.M3_GAMMAS]
    settings.append(Setting("REF", "REF_random_real", "none", 0.0))
    return settings


def kept_mask(setting: Setting, ranking: np.ndarray, random_ranking: np.ndarray) -> torch.Tensor:
    """
    保持する次元 (事後分布から取る次元) を表すマスクを作る．

    引数:
        setting (Setting): サンプリング設定．
        ranking (np.ndarray): 形状 (d,) の，属性関連度の降順に並べた次元の番号．
        random_ranking (np.ndarray): 形状 (d,) の，ランダムに並べた次元の番号 (M2r用)．
    戻り値:
        mask (torch.Tensor): 形状 (d,) の 0/1 のマスク (GPU上，float32)．
    """
    latent_dim = len(ranking)
    mask = np.zeros(latent_dim, dtype=np.float32)
    if setting.method == "M0":
        mask[:] = 1.0
    else:
        order = random_ranking if setting.method == "M2r" else ranking
        mask[order[:setting.k]] = 1.0
    return torch.from_numpy(mask).cuda()


def latent_coefficients(setting: Setting, mask: torch.Tensor, std: torch.Tensor):
    """
    潜在空間の方式について，f = a * mu + n * z の係数 a とノイズの標準偏差 n を求める．

    引数:
        setting (Setting): サンプリング設定．
        mask (torch.Tensor): 形状 (d,) の保持する次元のマスク．
        std (torch.Tensor): 形状 (N, d) の事後分布の標準偏差．
    戻り値:
        a (torch.Tensor): 形状 (d,) の mu に掛ける係数．
        n (torch.Tensor): 形状 (N, d) のノイズの標準偏差．
    """
    constant = (1.0 - setting.gamma ** 2) ** 0.5
    a = mask + (1.0 - mask) * setting.gamma
    n = mask * setting.alpha * std + (1.0 - mask) * constant
    return a, n


@torch.no_grad()
def sample(setting: Setting, model: torch.nn.Module, v: torch.Tensor, v0: torch.Tensor, mu: torch.Tensor,
           std: torch.Tensor, mask: torch.Tensor, generator: torch.Generator, other: torch.Tensor) -> torch.Tensor:
    """
    1つの設定について，全画像の匿名化後の顔ベクトルを K 本ずつサンプリングする．

    引数:
        setting (Setting): サンプリング設定．
        model (torch.nn.Module): 学習済みのVAE．
        v (torch.Tensor): 形状 (N, 512) の元の顔ベクトル．
        v0 (torch.Tensor): 形状 (N, 512) の decoder(mu)．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        mask (torch.Tensor): 形状 (d,) の保持する次元のマスク (潜在空間の方式でのみ使用)．
        generator (torch.Generator): 乱数生成器．
        other (torch.Tensor): 形状 (N, K) の，自分以外の画像の番号 (REFでのみ使用)．
    戻り値:
        v_tilde (torch.Tensor): 形状 (N, K, 512) の匿名化後の顔ベクトル．
    """
    num_images, num_samples = v.shape[0], common.NUM_SAMPLES
    if setting.method == "REF":
        return v[other]
    if setting.is_latent:
        a, n = latent_coefficients(setting, mask, std)
        z = torch.randn(num_images, num_samples, mu.shape[1], generator=generator, device="cuda")
        f = (a * mu).unsqueeze(1) + n.unsqueeze(1) * z
        return model.decode(f.reshape(-1, mu.shape[1])).reshape(num_images, num_samples, -1)

    base = v if setting.method == "M1" else v0
    epsilon = torch.randn(num_images, num_samples, base.shape[1], generator=generator, device="cuda")
    scale = setting.tau * base.norm(dim=1) / base.shape[1] ** 0.5
    return base.unsqueeze(1) + scale.view(-1, 1, 1) * epsilon


@torch.no_grad()
def closed_form(setting: Setting, model: torch.nn.Module, v: torch.Tensor, v0: torch.Tensor, mu: torch.Tensor,
                std: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    線形デコーダに基づき，cos(v_tilde, v)，cos(v_tilde, v0)，||v_tilde|| の期待値を近似する．

    潜在空間の方式では v_tilde = m + W (n * z)，m = W (a * mu) + b と書ける．ノイズの項は平均 0 で
    あり，その二乗ノルムの期待値は sum_j n_j^2 ||w_j||^2 である．ノイズと m，v，v0 との内積の
    期待値は 0 であるから，ノルムの揺らぎを無視すると

        cos(v_tilde, x) ≈ (m . x) / (||x|| * sqrt(||m||^2 + sum_j n_j^2 ||w_j||^2))

    となる．出力空間の方式では m は基準ベクトル，ノイズの二乗ノルムの期待値は tau^2 ||基準||^2 である．

    引数:
        setting (Setting): サンプリング設定．
        model，v，v0，mu，std，mask: `sample` と同じ．
    戻り値:
        prediction (dict): pred_cos_v，pred_cos_v0，pred_norm の，画像についての平均値．
    """
    if setting.method == "REF":
        return {"pred_cos_v": float("nan"), "pred_cos_v0": float("nan"), "pred_norm": float("nan")}
    if setting.is_latent:
        a, n = latent_coefficients(setting, mask, std)
        mean = model.decode(a * mu)
        column_sq_norm = model.decoder[0].weight.pow(2).sum(dim=0)
        noise_sq = (n.pow(2) * column_sq_norm).sum(dim=1)
    else:
        mean = v if setting.method == "M1" else v0
        noise_sq = (setting.tau ** 2) * mean.pow(2).sum(dim=1)

    total_norm = (mean.pow(2).sum(dim=1) + noise_sq).sqrt()
    return {
        "pred_cos_v": float(((mean * v).sum(dim=1) / (v.norm(dim=1) * total_norm)).mean()),
        "pred_cos_v0": float(((mean * v0).sum(dim=1) / (v0.norm(dim=1) * total_norm)).mean()),
        "pred_norm": float(total_norm.mean()),
    }
