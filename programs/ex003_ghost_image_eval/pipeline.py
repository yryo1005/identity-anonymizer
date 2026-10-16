"""
「ソースの顔ベクトル → GHOSTで顔画像を生成 → 生成画像を評価」という一連の処理をまとめたモジュール．
予稿の再現，試行，本実験で共通して用いる．
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import ghost
import settings
from classifiers import AttributePredictor, load_predictor
from evaluate import reidentification
from model import encode_all, load_model
from prepare import EvalData


@dataclass
class Context:
    """
    生成と評価に必要なモデルとデータを，選んだ画像についてまとめたもの．n は選んだ画像数である．

    属性:
        models (ghost.Models): 顔検出器，生成器，ArcFace，DEX．
        vae (torch.nn.Module): 学習済みのVAE．
        predictor (AttributePredictor): ex002 の線形の属性分類器 (正規化なし)．
        index (np.ndarray): 形状 (n,) の，評価用データの中での画像の番号．
        v (torch.Tensor): 形状 (n, 512) の元の顔ベクトル．
        mu, std (torch.Tensor): 形状 (n, d) の潜在変数の平均と標準偏差．
        v0 (torch.Tensor): 形状 (n, 512) の decoder(mu)．
        group (torch.Tensor): 形状 (n,) の重複グループの番号．
        gallery (torch.Tensor): 形状 (G, 512) の，L2正規化済みの探索対象の顔ベクトル．
        gallery_group (torch.Tensor): 形状 (G,) の探索対象の重複グループの番号．
        target224 (np.ndarray): 形状 (n, 224, 224, 3) の元の顔画像．
        resized256 (np.ndarray): 形状 (n, 256, 256, 3) の生成器のターゲット．
        original_dex (dict): 元の顔画像に対するDEXの予測 (各値は形状 (n,))．
        original_linear (dict): 元の顔ベクトルに対する線形分類器の予測 (各値は形状 (n,))．
        real_mean_norm (float): 学習用データの顔ベクトルの平均ノルム．
    """

    models: ghost.Models
    vae: torch.nn.Module
    predictor: AttributePredictor
    index: np.ndarray
    v: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    v0: torch.Tensor
    group: torch.Tensor
    gallery: torch.Tensor
    gallery_group: torch.Tensor
    target224: np.ndarray
    resized256: np.ndarray
    original_dex: Dict[str, np.ndarray]
    original_linear: Dict[str, np.ndarray]
    real_mean_norm: float


def batched_dex(models: ghost.Models, faces: np.ndarray) -> Dict[str, np.ndarray]:
    """
    多数の顔画像に対し，バッチに分けてDEXの予測を計算する．

    引数:
        models (ghost.Models): モデル一式．
        faces (np.ndarray): 形状 (n, H, W, 3) の顔画像 (BGR，uint8)．
    戻り値:
        prediction (dict): `ghost.dex_predict` と同じkeyを持ち，各値は形状 (n,) の配列．
    """
    parts = [ghost.dex_predict(models, np.asarray(faces[start:start + settings.BATCH_SIZE]))
             for start in range(0, len(faces), settings.BATCH_SIZE)]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def build_context(eval_data: EvalData, index: np.ndarray, models: Optional[ghost.Models] = None) -> Context:
    """
    選んだ画像について，生成と評価に必要なモデルとデータを準備する．

    引数:
        eval_data (EvalData): 評価用データ一式．
        index (np.ndarray): 形状 (n,) の，評価用データの中での画像の番号．
        models (ghost.Models または None): 読み込み済みのモデル．None の場合は新たに読み込む．
    戻り値:
        context (Context): 準備したモデルとデータ．
    """
    models = models or ghost.load_models()
    vae = load_model()
    predictor = load_predictor("raw")
    v = torch.from_numpy(eval_data.v[index]).cuda()
    mu, std = encode_all(vae, eval_data.v[index])
    with torch.no_grad():
        v0 = vae.decode(mu)
    target224 = np.asarray(eval_data.target224[index])
    return Context(
        models=models, vae=vae, predictor=predictor, index=index, v=v, mu=mu, std=std, v0=v0,
        group=torch.from_numpy(eval_data.group[index]).cuda(),
        gallery=F.normalize(torch.from_numpy(eval_data.gallery_v).cuda(), dim=1),
        gallery_group=torch.from_numpy(eval_data.gallery_group).cuda(),
        target224=target224, resized256=np.asarray(eval_data.resized256[index]),
        original_dex=batched_dex(models, target224),
        original_linear={name: value.cpu().numpy() for name, value in predictor.predict(v).items()},
        real_mean_norm=float(np.linalg.norm(eval_data.train_v, axis=1).mean()),
    )


@torch.no_grad()
def generate_and_evaluate(context: Context, sources: torch.Tensor, description: str = "generate"):
    """
    ソースの顔ベクトルから顔画像を生成し，画像ごとの評価指標を計算する．

    引数:
        context (Context): `build_context` で準備したモデルとデータ．
        sources (torch.Tensor): 形状 (n, 512) のソースの顔ベクトル．
        description (str): 進捗表示に用いる名前．
    戻り値:
        metrics (dict): 指標名をkey，形状 (n,) の配列をvalueとする辞書．
            cos_gen_v: 生成画像から再抽出した顔ベクトル v_gen と元の v のコサイン類似度，
            reid_top1 / reid_top10: v_gen をクエリとした再識別の成否，nn_cos: 最近傍の実ベクトルとの類似度，
            gen_norm: v_gen のノルム，dex_age / dex_is_female: 生成画像に対するDEXの予測，
            dex_gender_match: 元画像との性別の一致，dex_age_diff: 元画像との年齢の差 (符号付き)，
            lin_gender / lin_race / lin_age: v_gen に対する線形分類器の予測，
            lin_gender_agree / lin_race_agree: 元の v に対する線形分類器の予測との一致，
            lin_age_diff: 元の v に対する予測との差の絶対値．
        generated (np.ndarray): 形状 (n, 256, 256, 3) の生成した顔画像 (BGR，uint8)．
        v_gen (np.ndarray): 形状 (n, 512) の生成画像から再抽出した顔ベクトル．
    """
    num_images = len(context.index)
    images, embeddings, dex_parts = [], [], []
    for start in tqdm(range(0, num_images, settings.BATCH_SIZE), desc=description, leave=False):
        stop = start + settings.BATCH_SIZE
        generated = ghost.generate(context.models, context.resized256[start:stop], sources[start:stop])
        embeddings.append(ghost.embed(context.models, ghost.resize_to_crop(context.models, generated)))
        dex_parts.append(ghost.dex_predict(context.models, generated))
        images.append(generated)
    v_gen = torch.cat(embeddings)
    dex = {key: np.concatenate([part[key] for part in dex_parts]) for key in dex_parts[0]}

    reid = reidentification(v_gen.unsqueeze(1), context.gallery, context.gallery_group, context.group)
    linear = {name: value.cpu().numpy() for name, value in context.predictor.predict(v_gen).items()}
    metrics = {
        "cos_gen_v": F.cosine_similarity(v_gen, context.v, dim=1).cpu().numpy(),
        "reid_top1": reid["reid_top1"][:, 0].cpu().numpy(),
        "reid_top10": reid["reid_top10"][:, 0].cpu().numpy(),
        "nn_cos": reid["nn_cos"][:, 0].cpu().numpy(),
        "gen_norm": v_gen.norm(dim=1).cpu().numpy(),
        "dex_age": dex["age"],
        "dex_is_female": dex["is_female"].astype(np.float64),
        "dex_gender_match": (dex["is_female"] == context.original_dex["is_female"]).astype(np.float64),
        "dex_age_diff": dex["age"] - context.original_dex["age"],
        "lin_gender": linear["gender"].astype(np.float64),
        "lin_race": linear["race"].astype(np.float64),
        "lin_age": linear["age"],
        "lin_gender_agree": (linear["gender"] == context.original_linear["gender"]).astype(np.float64),
        "lin_race_agree": (linear["race"] == context.original_linear["race"]).astype(np.float64),
        "lin_age_diff": np.abs(linear["age"] - context.original_linear["age"]),
    }
    return metrics, np.concatenate(images), v_gen.cpu().numpy()


def summarize(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    画像ごとの指標を，設定ごとの要約値にまとめる．

    引数:
        metrics (dict): `generate_and_evaluate` の戻り値の指標．
    戻り値:
        summary (dict): 要約値の辞書．cos_gen_v (平均)，reid_top1，reid_top10，dex_gender_match (割合)，
            dex_age_abs_diff (差の絶対値の平均)，dex_age_rmse (予稿と同じ定義のRMSE)，dex_age_bias (差の平均)，
            lin_gender_agree，lin_race_agree，lin_age_diff，gen_norm，nn_cos など．
    """
    return {
        "num_images": int(len(metrics["cos_gen_v"])),
        "cos_gen_v": float(metrics["cos_gen_v"].mean()),
        "cos_gen_v_q05": float(np.quantile(metrics["cos_gen_v"], 0.05)),
        "cos_gen_v_q95": float(np.quantile(metrics["cos_gen_v"], 0.95)),
        "reid_top1": float(metrics["reid_top1"].mean()),
        "reid_top10": float(metrics["reid_top10"].mean()),
        "dex_gender_match": float(metrics["dex_gender_match"].mean()),
        "dex_age_abs_diff": float(np.abs(metrics["dex_age_diff"]).mean()),
        "dex_age_rmse": float(np.sqrt((metrics["dex_age_diff"] ** 2).mean())),
        "dex_age_bias": float(metrics["dex_age_diff"].mean()),
        "lin_gender_agree": float(metrics["lin_gender_agree"].mean()),
        "lin_race_agree": float(metrics["lin_race_agree"].mean()),
        "lin_age_diff": float(metrics["lin_age_diff"].mean()),
        "gen_norm": float(metrics["gen_norm"].mean()),
        "nn_cos": float(metrics["nn_cos"].mean()),
    }
