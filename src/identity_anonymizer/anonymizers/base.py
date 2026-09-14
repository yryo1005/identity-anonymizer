from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Anonymizer(nn.Module, ABC):
    """
    ArcFaceの顔ベクトルを匿名化するモデルの抽象基底クラス．

    GHOSTはソース顔として常にArcFaceの512次元埋め込みを要求するため，本インターフェースは
    「512次元の顔ベクトルを受け取り，同じ属性(性別・年齢等)を保持しつつ個人を特定する情報を
    変化させた512次元の顔ベクトルを返す」処理のみを共通化する．学習方法(教師なし再構成，
    敵対的学習等)はモデルごとに大きく異なりうるため，本クラスでは規定しない．

    新しい匿名化モデルを追加する場合は，本クラスを継承し `anonymize` を実装したうえで，
    `identity_anonymizer.anonymizers.registry.register_anonymizer` で登録すればよい．
    `FaceAnonymizerPipeline` 側の変更は不要である．
    """

    #: GHOST(ArcFace)が前提とする顔ベクトルの次元数．
    embedding_dim: int = 512

    @abstractmethod
    def anonymize(self, embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        顔ベクトルを匿名化する．

        引数:
            embedding (torch.Tensor): 形状 (B, embedding_dim) の元の顔ベクトル(ArcFace埋め込み)．
            **kwargs: モデル固有の実行時パラメータ(例: VAEにおける `noise_level`)．
        戻り値:
            anonymized_embedding (torch.Tensor): 形状 (B, embedding_dim) の匿名化後の顔ベクトル．
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, weight_path: str, map_location: str = "cpu", **init_kwargs) -> "Anonymizer":
        """
        学習済み重みからモデルをインスタンス化する．

        引数:
            weight_path (str): `torch.save(model.state_dict(), ...)` で保存された重みファイルのパス．
            map_location (str): 重みをロードするデバイス．
            **init_kwargs: モデルのコンストラクタに渡す追加引数．
        戻り値:
            model (Anonymizer): 重みを読み込んだモデルのインスタンス(評価モードに設定済み)．
        """
        model = cls(**init_kwargs)
        state_dict = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()
        return model
