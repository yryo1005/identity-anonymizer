import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.utils.seed import set_seed

AttributeDict = Dict[str, int]


class AttributeNNAnonymizer(Anonymizer):
    """
    性別・年齢・人種といった属性を入力とし，ArcFaceの顔ベクトルを近似的に出力するNNによる
    匿名化モデル．`references/` の予稿・発表資料で紹介されている「従来手法2」に相当する．

    `VAEAnonymizer` がターゲットの顔画像から計算したArcFace顔ベクトルそのものを入力とするのに
    対し，本モデルはターゲットの顔画像を必要とせず，属性のみから顔ベクトルを生成する．
    ただし，これは同時に「ターゲットの属性を事前に取得しておく必要がある」という，予稿でも
    指摘されている従来手法の課題でもある(`Anonymizer` の入力を可変にできることを示す例として，
    また `VAEAnonymizer` との設計上のトレードオフを示す例として実装している)．

    学習は，UTKFaceの各顔画像 $ X_n $ とその属性 $ I_n $ の組を用いて，
    $ \\arg\\min_{NN} \\sum_n \\| \\mathrm{ArcFace}(X_n) - NN(I_n) \\|_2 $ を最小化することで行う
    (`identity_anonymizer.anonymizers.attribute_nn.train_attribute_nn_anonymizer` を参照)．
    """

    #: UTKFaceの人種ラベルのカテゴリ数(0: 白人，1: 黒人，2: アジア人，3: インド人，4: その他)．
    NUM_RACE_CATEGORIES = 5
    #: 年齢を [0, 1] へ正規化する際に用いる最大値．
    MAX_AGE = 116.0

    def __init__(self, hidden_dim: int = 256, seed: int = 0):
        """
        引数:
            hidden_dim (int): 中間層のユニット数．
            seed (int): パラメータ初期化に用いるseed値．
        戻り値:
            なし．
        """
        super().__init__()

        set_seed(seed=seed)

        # 入力特徴量: 年齢(1次元，正規化済み) + 性別のone-hot(2次元) + 人種のone-hot(5次元)
        input_dim = 1 + 2 + self.NUM_RACE_CATEGORIES
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def validate_input(self, x: Union[AttributeDict, List[AttributeDict]]) -> None:
        """
        `Anonymizer` インターフェースの実装．入力が，キー `"age"`，`"gender"`，`"race"` を
        持つdict，またはそのdictのlist(バッチ)であることを検証する．

        引数:
            x (dict または list[dict]): `anonymize` に渡された入力．
        戻り値:
            なし．
        """
        attributes_list = x if isinstance(x, list) else [x]
        if len(attributes_list) == 0:
            raise ValueError("AttributeNNAnonymizerの入力(属性のリスト)が空です．")

        for attributes in attributes_list:
            if not isinstance(attributes, dict):
                raise TypeError(
                    f"AttributeNNAnonymizerの入力はdict，またはdictのlistである必要があります: {type(attributes)}"
                )
            for key in ("age", "gender", "race"):
                if key not in attributes:
                    raise ValueError(f"属性 '{key}' が入力に含まれていません: {attributes}")
            if attributes["gender"] not in (0, 1):
                raise ValueError(f"genderは0(男性)または1(女性)である必要があります: {attributes['gender']}")
            if not (0 <= attributes["race"] < self.NUM_RACE_CATEGORIES):
                raise ValueError(
                    f"raceは0〜{self.NUM_RACE_CATEGORIES - 1}の整数である必要があります: {attributes['race']}"
                )

    def _encode_attributes(self, attributes_list: List[AttributeDict]) -> torch.Tensor:
        """
        属性のdictのリストを，NNへ入力するための特徴量テンソルへ変換する．

        引数:
            attributes_list (list[dict]): `validate_input` を通過した属性のリスト．
        戻り値:
            features (torch.Tensor): 形状 (B, 1 + 2 + NUM_RACE_CATEGORIES) の特徴量．
        """
        rows = []
        for attributes in attributes_list:
            age_feature = [min(float(attributes["age"]), self.MAX_AGE) / self.MAX_AGE]
            gender_onehot = F.one_hot(torch.tensor(attributes["gender"]), num_classes=2).float().tolist()
            race_onehot = F.one_hot(
                torch.tensor(attributes["race"]), num_classes=self.NUM_RACE_CATEGORIES
            ).float().tolist()
            rows.append(age_feature + gender_onehot + race_onehot)
        return torch.tensor(rows, dtype=torch.float32)

    def _anonymize(self, x: Union[AttributeDict, List[AttributeDict]], **kwargs) -> torch.Tensor:
        """
        `Anonymizer` インターフェースの実装．属性から顔ベクトルを生成する．

        引数:
            x (dict または list[dict]): `validate_input` 済みの属性(単体，またはバッチ分のlist)．
            **kwargs: 未使用(他のAnonymizerとインターフェースを揃えるために受け取る)．
        戻り値:
            embedding (torch.Tensor): 形状 (B, output_dim) の顔ベクトル．
        """
        attributes_list = x if isinstance(x, list) else [x]
        device = next(self.parameters()).device
        features = self._encode_attributes(attributes_list).to(device)
        return self.net(features)


def train_attribute_nn_anonymizer(
    target_weight_path: str,
    attributes: np.ndarray,
    embeddings: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_dim: int = 256,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    `AttributeNNAnonymizer` を学習する．

    目的関数は，UTKFaceの各画像のArcFace顔ベクトルと，NNが属性から予測した顔ベクトルとの
    L2ノルムの平均である:
    $ \\arg\\min_{NN} \\frac{1}{N} \\sum_n \\| \\mathrm{ArcFace}(X_n) - NN(I_n) \\|_2 $

    引数:
        target_weight_path (str): 学習済み重みの保存先パス．
        attributes (np.ndarray): 形状 (N, 3) の属性(列の順に age, gender, race)．
        embeddings (np.ndarray): 形状 (N, 512) のArcFace顔ベクトル．
        train_indices (np.ndarray): 学習用データのインデックス．
        test_indices (np.ndarray): 検証用データのインデックス．
        epochs (int): エポック数．
        batch_size (int): バッチサイズ．
        hidden_dim (int): 中間層のユニット数．
        seed (int): 再現性確保のためのseed値．
        verbose (bool): 学習の進捗を表示するかどうか．
    戻り値:
        history (dict): 各エポックの学習/検証のL2ノルム損失を記録した辞書．
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_attribute_dicts(rows: np.ndarray) -> List[AttributeDict]:
        return [{"age": int(row[0]), "gender": int(row[1]), "race": int(row[2])} for row in rows]

    train_attr_dicts = to_attribute_dicts(attributes[train_indices])
    test_attr_dicts = to_attribute_dicts(attributes[test_indices])

    model = AttributeNNAnonymizer(hidden_dim=hidden_dim, seed=seed).to(device)

    train_features = model._encode_attributes(train_attr_dicts)
    test_features = model._encode_attributes(test_attr_dicts)
    train_targets = torch.tensor(embeddings[train_indices], dtype=torch.float32)
    test_targets = torch.tensor(embeddings[test_indices], dtype=torch.float32)

    set_seed(seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True, generator=generator,
    )
    test_loader = DataLoader(TensorDataset(test_features, test_targets), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def loss_function(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(pred - target, dim=1).mean()

    def epoch(dataloader: DataLoader, optimizer_: Optional[torch.optim.Optimizer] = None) -> float:
        total_loss = 0.0
        num_samples = 0
        pb = tqdm(dataloader, leave=False) if verbose else dataloader
        for features, targets in pb:
            features, targets = features.to(device), targets.to(device)
            if optimizer_ is not None:
                optimizer_.zero_grad()
                pred = model.net(features)
                loss = loss_function(pred, targets)
                loss.backward()
                optimizer_.step()
            else:
                with torch.no_grad():
                    pred = model.net(features)
                    loss = loss_function(pred, targets)
            total_loss += loss.item() * len(features)
            num_samples += len(features)
            if verbose:
                pb.set_postfix({"loss": total_loss / num_samples})
        return total_loss / num_samples

    history = {"train_loss": [], "test_loss": []}
    best_test_loss = float("inf")
    os.makedirs(os.path.dirname(target_weight_path) or ".", exist_ok=True)

    for i in range(epochs + 1):
        if i == 0:
            model.eval()
            train_loss = epoch(train_loader, optimizer_=None)
        else:
            model.train()
            train_loss = epoch(train_loader, optimizer_=optimizer)

        model.eval()
        test_loss = epoch(test_loader, optimizer_=None)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if verbose:
            print(f"epoch {i}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), target_weight_path)

    return history
