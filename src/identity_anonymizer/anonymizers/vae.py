from typing import Tuple

import torch
import torch.nn as nn

from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.utils.seed import set_seed


class VAEAnonymizer(Anonymizer):
    """
    Variational Autoencoder(VAE)による顔ベクトルの匿名化モデル．

    ArcFaceの顔ベクトル(512次元)を入力とし，潜在空間へエンコードし，再パラメータ化トリックで
    サンプリングした後デコードすることで，同じ属性(性別・年齢等のマクロな情報)を保持しつつ，
    個人の特定に直結するミクロな情報のみを変化させた顔ベクトルを生成する．`noise_level` に
    よってサンプリング時の標準偏差をスケーリングし，匿名化の強度(元の顔との類似度とのトレード
    オフ)を制御する．
    """

    #: このモデルが入力として要求するArcFace顔ベクトルの次元数．
    input_dim: int = 512

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
            nn.Linear(self.input_dim, latent_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.output_dim),
        )

    def validate_input(self, x: torch.Tensor) -> None:
        """
        `Anonymizer` インターフェースの実装．入力がArcFace顔ベクトルの形状 (B, input_dim) の
        `torch.Tensor` であることを検証する．

        引数:
            x (torch.Tensor): `anonymize` に渡された入力．
        戻り値:
            なし．
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"VAEAnonymizerの入力はtorch.Tensorである必要がありますが，{type(x)} が渡されました．")
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"VAEAnonymizerの入力形状が不正です: {tuple(x.shape)} "
                f"(期待する形状: (B, {self.input_dim}))．"
            )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        顔ベクトルを潜在変数の平均・分散の対数へエンコードする．

        引数:
            x (torch.Tensor): 形状 (B, input_dim) の顔ベクトル．
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
            v_hat (torch.Tensor): 形状 (B, output_dim) のデコード結果．
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, noise_level: float = 1.0):
        """
        学習・評価の両方で使用する順伝播．エンコード・サンプリング・デコードを行う．

        引数:
            x (torch.Tensor): 形状 (B, input_dim) の入力顔ベクトル．
            noise_level (float): サンプリング時の標準偏差に乗じるスケール係数．
        戻り値:
            v_hat (torch.Tensor): 形状 (B, output_dim) の再構成(匿名化後)の顔ベクトル．
            mu (torch.Tensor): 形状 (B, latent_dim) の潜在変数の平均(損失計算用)．
            logvar (torch.Tensor): 形状 (B, latent_dim) の潜在変数の分散の対数(損失計算用)．
        """
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar, noise_level)
        return self.decode(z), mu, logvar

    def _anonymize(self, x: torch.Tensor, noise_level: float = 1.0) -> torch.Tensor:
        """
        `Anonymizer` インターフェースの実装．顔ベクトルを匿名化する．

        引数:
            x (torch.Tensor): 形状 (B, input_dim) の元の顔ベクトル(`validate_input` 済み)．
            noise_level (float): VAEのサンプリングに用いるスケール係数．大きいほど匿名性が
                高まる一方，元の属性との一致度は低下する．
        戻り値:
            anonymized_embedding (torch.Tensor): 形状 (B, output_dim) の匿名化後の顔ベクトル．
        """
        v_hat, _, _ = self.forward(x, noise_level=noise_level)
        return v_hat
