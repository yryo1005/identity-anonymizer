from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Anonymizer(nn.Module, ABC):
    """
    匿名化後の顔ベクトルを生成するモデルの抽象基底クラス．

    GHOSTはソース顔として常にArcFaceの512次元埋め込みを要求するため，**出力**は必ず
    形状 (B, output_dim) の顔ベクトルでなければならない．一方，**入力**の形式・形状は
    サブクラスに委ねる．`references/` の予稿・発表資料で紹介されている従来手法のように，
    顔ベクトルではなく性別・年齢・人種といった属性を入力とする実装も考えられるため，
    入力の型はモデルごとに異なってよい(`VAEAnonymizer` はArcFaceの顔ベクトルそのものを
    入力とする)．学習方法(教師なし再構成，敵対的学習等)についても同様にモデルごとに
    大きく異なりうるため，本クラスでは規定しない．

    新しい匿名化モデルを追加する場合は，本クラスを継承し，実際の変換処理を行う
    `_anonymize` と，入力の形式・形状を検証する `validate_input` を実装したうえで，
    `identity_anonymizer.anonymizers.registry.register_anonymizer` で登録すればよい．
    `anonymize` は基底クラスが提供するテンプレートメソッドであり，`validate_input` に
    よる入力検証と，`_anonymize` の出力形状の検証を行う．サブクラス側で上書きしない．
    `FaceAnonymizerPipeline` 側の変更も不要である．
    """

    #: GHOST(ArcFace)が前提とする，出力される匿名化後の顔ベクトルの次元数．
    #: 入力の次元・形式はサブクラスごとに異なりうる(`validate_input` を参照)．
    output_dim: int = 512

    def anonymize(self, x: Any, **kwargs) -> torch.Tensor:
        """
        入力を検証したうえで，匿名化後の顔ベクトルを生成する．サブクラスでは上書きせず，
        代わりに `_anonymize` と `validate_input` を実装すること．

        引数:
            x (Any): 匿名化のための入力．具体的な形式・形状はサブクラスの `validate_input`
                が規定する(例: `VAEAnonymizer` では形状 (B, 512) のArcFace顔ベクトル)．
            **kwargs: モデル固有の実行時パラメータ(例: VAEにおける `noise_level`)．
        戻り値:
            anonymized_embedding (torch.Tensor): 形状 (B, output_dim) の匿名化後の顔ベクトル．
        """
        self.validate_input(x)
        output = self._anonymize(x, **kwargs)

        if output.dim() != 2 or output.shape[-1] != self.output_dim:
            raise ValueError(
                f"{type(self).__name__}._anonymize の出力形状が不正です: {tuple(output.shape)} "
                f"(期待する形状: (B, {self.output_dim}))．"
            )
        return output

    @abstractmethod
    def _anonymize(self, x: Any, **kwargs) -> torch.Tensor:
        """
        実際の匿名化処理．`anonymize` (検証込みのテンプレートメソッド)から呼び出される．

        引数:
            x (Any): `validate_input` による検証を通過した入力．
            **kwargs: モデル固有の実行時パラメータ．
        戻り値:
            anonymized_embedding (torch.Tensor): 形状 (B, output_dim) の匿名化後の顔ベクトル．
        """
        raise NotImplementedError

    @abstractmethod
    def validate_input(self, x: Any) -> None:
        """
        入力 `x` の型・形式・形状が，このモデルの想定と一致するかを検証する．

        引数:
            x (Any): `anonymize` に渡された入力．
        戻り値:
            なし．想定と一致しない場合は `TypeError` または `ValueError` を送出すること．
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
