"""
評価用の属性分類器．

実際のArcFaceの顔ベクトル v から，性別 (ロジスティック回帰)，人種 (多クラスロジスティック回帰)，
年齢 (リッジ回帰) を予測する線形モデルを学習用データで学習する．学習したモデルは，標準化を
重みに畳み込んだ線形写像 (W, b) として保存し，生成した顔ベクトルにGPU上で適用できるようにする．

GHOSTはソースの埋め込みを正規化せずに用いるため，正規化しないベクトルで学習したもの ("raw") を
主に用いる．L2正規化したベクトルで学習したもの ("l2") は参考として併せて保存する．

    python programs/ex002_partial_prior_sampling/classifiers.py
"""

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import common
from data import Dataset, load_dataset
from model import encode_all, load_model

METHOD = "classifiers"
VARIANTS = ["raw", "l2"]


@dataclass
class AttributePredictor:
    """
    線形の属性予測器．入力の前処理 (L2正規化の有無) と，標準化を畳み込んだ重みを保持する．

    属性:
        normalize (bool): 入力をL2正規化するかどうか．
        weights (dict): "gender" (形状 (1, 512))，"race" (形状 (5, 512))，"age" (形状 (1, 512)) の重み．
        biases (dict): 各属性のバイアス (形状はそれぞれ (1,)，(5,)，(1,))．
    """

    normalize: bool
    weights: Dict[str, torch.Tensor]
    biases: Dict[str, torch.Tensor]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        顔ベクトルから属性を予測する．

        引数:
            x (torch.Tensor): 形状 (M, 512) の顔ベクトル (GPU上)．
        戻り値:
            prediction (dict): "gender" (形状 (M,)，0か1)，"race" (形状 (M,)，0〜4)，
                "age" (形状 (M,)，実数) の予測値．
        """
        x = x.double()
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=1)
        scores = {name: x @ self.weights[name].T + self.biases[name] for name in self.weights}
        return {
            "gender": (scores["gender"][:, 0] > 0).long(),
            "race": scores["race"].argmax(dim=1),
            "age": scores["age"][:, 0],
        }


def fold_scaler(scaler: StandardScaler, coef: np.ndarray, intercept: np.ndarray):
    """
    標準化と線形モデルを合成し，元の入力に直接適用できる重みとバイアスに変換する．

    引数:
        scaler (StandardScaler): 学習済みの標準化器．
        coef (np.ndarray): 形状 (C, 512) の，標準化後の入力に対する重み．
        intercept (np.ndarray): 形状 (C,) のバイアス．
    戻り値:
        weight (np.ndarray): 形状 (C, 512) の，元の入力に対する重み．
        bias (np.ndarray): 形状 (C,) の，元の入力に対するバイアス．
    """
    weight = coef / scaler.scale_
    return weight, intercept - weight @ scaler.mean_


def train_variant(dataset: Dataset, normalize: bool) -> Dict[str, np.ndarray]:
    """
    1つの前処理について，3つの属性の線形モデルを学習用データで学習する．

    引数:
        dataset (Dataset): ラベルと分割を含むデータ一式．
        normalize (bool): 入力をL2正規化するかどうか．
    戻り値:
        params (dict): `{属性}_weight`，`{属性}_bias` をkeyとする重みの辞書．
    """
    x = dataset.v.astype(np.float64)
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    train = ~dataset.is_eval
    scaler = StandardScaler().fit(x[train])
    x_train = scaler.transform(x[train])

    gender = LogisticRegression(max_iter=3000).fit(x_train, dataset.gender[train])
    race = LogisticRegression(max_iter=3000).fit(x_train, dataset.race[train])
    age = Ridge(alpha=1.0).fit(x_train, dataset.age[train])

    params = {}
    fitted = {"gender": (gender.coef_, gender.intercept_), "race": (race.coef_, race.intercept_),
              "age": (age.coef_[None, :], np.array([age.intercept_]))}
    for name, (coef, intercept) in fitted.items():
        params[f"{name}_weight"], params[f"{name}_bias"] = fold_scaler(scaler, coef, intercept)
    return params


def load_predictor(variant: str = "raw") -> AttributePredictor:
    """
    保存済みの属性予測器を読み込む．

    引数:
        variant (str): "raw" (正規化なし) または "l2" (L2正規化あり)．
    戻り値:
        predictor (AttributePredictor): GPU上の属性予測器．
    """
    params = np.load(os.path.join(common.OUTPUT_DIR, METHOD, variant, "0", "weights.npz"))
    names = ["gender", "race", "age"]
    return AttributePredictor(
        normalize=(variant == "l2"),
        weights={name: torch.from_numpy(params[f"{name}_weight"]).cuda() for name in names},
        biases={name: torch.from_numpy(params[f"{name}_bias"]).cuda() for name in names},
    )


def evaluate(predictor: AttributePredictor, x: torch.Tensor, dataset: Dataset, mask: np.ndarray) -> Dict[str, float]:
    """
    属性予測器の精度を，指定した画像について計算する．

    引数:
        predictor (AttributePredictor): 属性予測器．
        x (torch.Tensor): 形状 (N, 512) の入力ベクトル．
        dataset (Dataset): ラベルを含むデータ一式．
        mask (np.ndarray): 形状 (N,) のbool配列．評価対象の画像が True．
    戻り値:
        accuracy (dict): gender_acc，race_acc，age_mae の辞書．
    """
    prediction = {name: value.cpu().numpy()[mask] for name, value in predictor.predict(x).items()}
    return {
        "gender_acc": float((prediction["gender"] == dataset.gender[mask]).mean()),
        "race_acc": float((prediction["race"] == dataset.race[mask]).mean()),
        "age_mae": float(np.abs(prediction["age"] - dataset.age[mask]).mean()),
    }


def run() -> None:
    """
    2つの前処理について属性予測器を学習し，重みと，学習用・評価用データでの精度を保存する．
    あわせて，VAEの再構成 v0 = decoder(mu) に適用したときの精度も記録する．

    引数:
        なし．
    戻り値:
        なし．
    """
    dataset = load_dataset()
    model = load_model()
    v = torch.from_numpy(dataset.v).cuda()
    with torch.no_grad():
        v0 = model.decode(encode_all(model, dataset.v)[0])

    for variant in VARIANTS:
        target_dir = common.condition_dir(METHOD, variant, 0)
        if common.is_finished(target_dir):
            print(f"実行済みのためスキップ: {target_dir}")
            continue
        np.savez(os.path.join(target_dir, "weights.npz"), **train_variant(dataset, normalize=(variant == "l2")))
        predictor = load_predictor(variant)
        result = {
            "variant": variant,
            "num_train": int((~dataset.is_eval).sum()), "num_eval": int(dataset.is_eval.sum()),
            "train_on_v": evaluate(predictor, v, dataset, ~dataset.is_eval),
            "eval_on_v": evaluate(predictor, v, dataset, dataset.is_eval),
            "eval_on_v0": evaluate(predictor, v0, dataset, dataset.is_eval),
        }
        common.save_json(result, os.path.join(target_dir, "meta.json"))
        print(result)


if __name__ == "__main__":
    run()
