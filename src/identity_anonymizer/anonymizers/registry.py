from typing import Dict, Type

from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.anonymizers.vae import VAEAnonymizer

_ANONYMIZER_REGISTRY: Dict[str, Type[Anonymizer]] = {
    "vae": VAEAnonymizer,
}


def register_anonymizer(name: str, anonymizer_cls: Type[Anonymizer]) -> None:
    """
    新しい匿名化モデルをレジストリへ登録する．

    VAE以外の匿名化モデルを追加する場合は，`Anonymizer` を継承したクラスを実装したうえで
    本関数を呼び出せば，`get_anonymizer(name)` から利用できるようになる．
    `FaceAnonymizerPipeline` 等の呼び出し側のコードを変更する必要はない．

    引数:
        name (str): モデルを参照するための識別名．
        anonymizer_cls (Type[Anonymizer]): `Anonymizer` を継承したクラス．
    戻り値:
        なし．
    """
    if not issubclass(anonymizer_cls, Anonymizer):
        raise TypeError(f"{anonymizer_cls} は Anonymizer のサブクラスである必要があります．")
    _ANONYMIZER_REGISTRY[name] = anonymizer_cls


def get_anonymizer(name: str, weight_path: str = None, map_location: str = "cpu", **init_kwargs) -> Anonymizer:
    """
    登録済みの匿名化モデルをインスタンス化する．

    引数:
        name (str): レジストリに登録されたモデルの識別名(例: "vae")．
        weight_path (str または None): 学習済み重みのパス．指定した場合は
            `Anonymizer.from_pretrained` で重みを読み込む．None の場合は初期化のみ行う．
        map_location (str): 重みをロードするデバイス(weight_path指定時のみ使用)．
        **init_kwargs: モデルのコンストラクタに渡す追加引数．
    戻り値:
        model (Anonymizer): インスタンス化された匿名化モデル．
    """
    if name not in _ANONYMIZER_REGISTRY:
        available = ", ".join(sorted(_ANONYMIZER_REGISTRY.keys()))
        raise KeyError(f"未登録のAnonymizerです: '{name}'．登録済み: [{available}]")

    anonymizer_cls = _ANONYMIZER_REGISTRY[name]

    if weight_path is not None:
        return anonymizer_cls.from_pretrained(weight_path, map_location=map_location, **init_kwargs)
    return anonymizer_cls(**init_kwargs)


def list_anonymizers() -> list:
    """
    登録済みの匿名化モデルの識別名の一覧を取得する．

    引数:
        なし．
    戻り値:
        names (list[str]): 登録済みモデルの識別名のリスト．
    """
    return sorted(_ANONYMIZER_REGISTRY.keys())
