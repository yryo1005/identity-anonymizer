import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch

from identity_anonymizer import config
from identity_anonymizer.evaluation.dex_models import DexAgeModel, DexGenderModel


@dataclass
class DexModels:
    """DEXの年齢・性別推定モデルをまとめて保持するデータクラス．"""

    age_model: DexAgeModel
    gender_model: DexGenderModel


def load_dex_models(weights_dir: str = None, device: str = "cpu") -> DexModels:
    """
    DEXの年齢・性別推定モデルを読み込む．

    引数:
        weights_dir (str または None): `age_sd.pth`, `gender_sd.pth` を格納したディレクトリ．
            None の場合は `identity_anonymizer.config.DEX_WEIGHTS_DIR` を使用する．
        device (str): モデルを配置するデバイス．
    戻り値:
        models (DexModels): 読み込み済みの年齢・性別推定モデル．
    """
    if weights_dir is None:
        weights_dir = config.DEX_WEIGHTS_DIR

    age_model = DexAgeModel()
    age_model.load_state_dict(torch.load(os.path.join(weights_dir, "age_sd.pth"), map_location=device))
    age_model.eval()
    age_model = age_model.to(device)

    gender_model = DexGenderModel()
    gender_model.load_state_dict(torch.load(os.path.join(weights_dir, "gender_sd.pth"), map_location=device))
    gender_model.eval()
    gender_model = gender_model.to(device)

    return DexModels(age_model=age_model, gender_model=gender_model)


def _preprocess(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    DEXモデルへ入力するための前処理(224x224へのリサイズとCHW化)を行う．

    引数:
        image_bgr (np.ndarray): 形状 (H, W, 3) の顔画像(BGR，任意サイズ)．
        device (torch.device): テンソルを配置するデバイス．
    戻り値:
        tensor (torch.Tensor): 形状 (1, 3, 224, 224) の入力テンソル(0〜255のfloat，正規化なし)．
    """
    image = cv2.resize(image_bgr, (224, 224))
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    return torch.from_numpy(image).float().to(device)


def estimate_age(image_bgr: np.ndarray, models: DexModels) -> float:
    """
    顔画像から見た目年齢を推定する．推論に用いるデバイスは `models.age_model` のデバイスに従う．

    引数:
        image_bgr (np.ndarray): 形状 (H, W, 3) の顔画像(BGR)．
        models (DexModels): `load_dex_models` で読み込んだモデル一式．
    戻り値:
        age (float): 推定された見た目年齢(0〜100歳の期待値)．
    """
    device = next(models.age_model.parameters()).device
    tensor = _preprocess(image_bgr, device)
    with torch.no_grad():
        probs = models.age_model(tensor).cpu().numpy().squeeze()
    return float(sum((i + 1) * p for i, p in enumerate(probs)))


def estimate_gender(image_bgr: np.ndarray, models: DexModels) -> Tuple[float, float]:
    """
    顔画像から性別を推定する．推論に用いるデバイスは `models.gender_model` のデバイスに従う．

    引数:
        image_bgr (np.ndarray): 形状 (H, W, 3) の顔画像(BGR)．
        models (DexModels): `load_dex_models` で読み込んだモデル一式．
    戻り値:
        (female_prob, male_prob) (Tuple[float, float]): 女性・男性それぞれの確率．
    """
    device = next(models.gender_model.parameters()).device
    tensor = _preprocess(image_bgr, device)
    with torch.no_grad():
        probs = models.gender_model(tensor).cpu().numpy().squeeze()
    return float(probs[0]), float(probs[1])
