from typing import Tuple

import torch
import torch.nn as nn

from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.utils.seed import set_seed


class VAEAnonymizer(Anonymizer):
    """
    Variational Autoencoder(VAE)による顔ベクトルの匿名化モデル．

    ArcFaceの顔ベクトル(512次元)を潜在空間へエンコードし，再パラメータ化トリックでサンプリング
    した後デコードすることで，同じ属性(性別・年齢等のマクロな情報)を保持しつつ，個人の特定に
    直結するミクロな情報のみを変化させた顔ベクトルを生成する．`noise_level` によって
    サンプリング時の標準偏差をスケーリングし，匿名化の強度(元の顔との類似度とのトレードオフ)を
    制御する．
    """

    def __init__(self, latent_dim: int = 128, seed: int = 0):
        """
        引数:
            latent_dim (int): 潜在空間の次元数．
            seed (int): パラメータ初期化に用いるseed値．
        戻り値:
            なし．
        """
        super().__init__()

        set_seed(seed=seed)

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, latent_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        顔ベクトルを潜在変数の平均・分散の対数へエンコードする．

        引数:
            x (torch.Tensor): 形状 (B, embedding_dim) の顔ベクトル．
        戻り値:
            mu (torch.Tensor): 形状 (B, latent_dim) の潜在変数の平均．
            logvar (torch.Tensor): 形状 (B, latent_dim) の潜在変数の分散の対数．
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, noise_level: float) -> torch.Tensor:
        """
        再パラメータ化トリックにより潜在変数をサンプリングする．

        引数:
            mu (torch.Tensor): 形状 (B, latent_dim) の潜在変数の平均．
            logvar (torch.Tensor): 形状 (B, latent_dim) の潜在変数の分散の対数．
            noise_level (float): サンプリング時の標準偏差に乗じるスケール係数．
        戻り値:
            z (torch.Tensor): 形状 (B, latent_dim) のサンプリングされた潜在変数．
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * noise_level

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        潜在変数を顔ベクトルへデコードする．

        引数:
            z (torch.Tensor): 形状 (B, latent_dim) の潜在変数．
        戻り値:
            v_hat (torch.Tensor): 形状 (B, embedding_dim) のデコード結果．
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, noise_level: float = 1.0):
        """
        学習・評価の両方で使用する順伝播．エンコード・サンプリング・デコードを行う．

        引数:
            x (torch.Tensor): 形状 (B, embedding_dim) の入力顔ベクトル．
            noise_level (float): サンプリング時の標準偏差に乗じるスケール係数．
        戻り値:
            v_hat (torch.Tensor): 形状 (B, embedding_dim) の再構成(匿名化後)の顔ベクトル．
            mu (torch.Tensor): 形状 (B, latent_dim) の潜在変数の平均(損失計算用)．
            logvar (torch.Tensor): 形状 (B, latent_dim) の潜在変数の分散の対数(損失計算用)．
        """
        mu, logvar = self.encode(x.view(-1, self.embedding_dim))
        z = self.reparameterize(mu, logvar, noise_level)
        return self.decode(z), mu, logvar

    def anonymize(self, embedding: torch.Tensor, noise_level: float = 1.0) -> torch.Tensor:
        """
        `Anonymizer` インターフェースの実装．顔ベクトルを匿名化する．

        引数:
            embedding (torch.Tensor): 形状 (B, embedding_dim) の元の顔ベクトル．
            noise_level (float): VAEのサンプリングに用いるスケール係数．大きいほど匿名性が
                高まる一方，元の属性との一致度は低下する．
        戻り値:
            anonymized_embedding (torch.Tensor): 形状 (B, embedding_dim) の匿名化後の顔ベクトル．
        """
        v_hat, _, _ = self.forward(embedding, noise_level=noise_level)
        return v_hat
