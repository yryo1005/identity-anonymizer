"""
貼り戻した画像での評価．

既存の `FaceAnonymizerPipeline.anonymize_image` をそのまま用いて，生成した顔を元画像へ貼り戻した
画像を作り，DEXへの入力を次の3通りに変えて属性の一致を評価する．

    crop   : 切り出した顔 同士 (元画像から切り出した顔 と 生成器が出力した顔)．既存の評価処理と同じ．
    full   : 画像の全体 同士 (元画像の全体 と 貼り戻した画像の全体)．
    recrop : 元画像から切り出した顔 と 貼り戻した画像から改めて検出・切り出した顔．

貼り戻しにはランドマーク検出器 (mxnet) が必要なため，`env_mxcpu.sh` の環境で実行する．
ソースの顔ベクトルは，あらかじめ決めたベクトルを返す `Anonymizer` 経由で渡す．これにより，
ベクトル空間で作ったものと同じ顔ベクトルを，既存のパイプラインに変更を加えずに入力できる．

    source programs/ex003_ghost_image_eval/env_mxcpu.sh
    $PYTHON programs/ex003_ghost_image_eval/pasteback.py --target reproduction
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import ghost
import pipeline
import prepare
import samplers
import settings
from evaluate import reidentification
from identity_anonymizer.anonymizers.base import Anonymizer
from machine_learning_utils import set_seed

MODES = ["crop", "full", "recrop"]


class FixedVectorAnonymizer(Anonymizer):
    """
    あらかじめ設定した顔ベクトルをそのまま返す `Anonymizer`．`vector` が None の場合は入力をそのまま返す
    (自分自身との顔交換)．既存のパイプラインに，外部で作ったソースの顔ベクトルを渡すために用いる．
    """

    def __init__(self):
        """
        引数:
            なし．
        戻り値:
            なし．
        """
        super().__init__()
        self.vector: Optional[torch.Tensor] = None

    def validate_input(self, x: torch.Tensor) -> None:
        """
        入力が形状 (B, 512) の `torch.Tensor` であることを検証する．

        引数:
            x (torch.Tensor): `anonymize` に渡された入力．
        戻り値:
            なし．
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[-1] != self.output_dim:
            raise ValueError("入力は形状 (B, 512) の torch.Tensor である必要があります．")

    def _anonymize(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        設定済みの顔ベクトルを返す．

        引数:
            x (torch.Tensor): 形状 (1, 512) のターゲットの顔ベクトル．
        戻り値:
            vector (torch.Tensor): 形状 (1, 512) のソースの顔ベクトル．
        """
        if self.vector is None:
            return x
        return self.vector.to(device=x.device, dtype=x.dtype).view(1, -1)


def build_pipeline(models: ghost.Models):
    """
    既存の `FaceAnonymizerPipeline` を，ランドマーク検出器をCPUに配置して組み立てる．
    既存の `load_ghost_models` はランドマーク検出器をGPUで初期化するため，CPU版の mxnet では使えない．
    その他のモデルの設定は `load_ghost_models` と同じである．

    引数:
        models (ghost.Models): 顔検出器，生成器，ArcFace，DEX．
    戻り値:
        face_pipeline (FaceAnonymizerPipeline): 既存のパイプライン．
        anonymizer (FixedVectorAnonymizer): パイプラインに組み込んだ `Anonymizer`．
    """
    from coordinate_reg.image_infer import Handler
    from identity_anonymizer import config
    from identity_anonymizer.faceswap.models import GhostModels
    from identity_anonymizer.faceswap.pipeline import FaceAnonymizerPipeline

    weights_dir = config.GHOST_WEIGHTS_DIR
    handler = Handler(os.path.join(weights_dir, "2d106det"), 0, ctx_id=-1, det_size=640, root=weights_dir)
    ghost_models = GhostModels(app=models.app, generator=models.generator, arcface=models.arcface,
                               handler=handler, super_resolution=None)
    anonymizer = FixedVectorAnonymizer()
    return FaceAnonymizerPipeline(ghost_models, anonymizer), anonymizer


def reproduction_sources(context: pipeline.Context) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """
    report_003 の予稿の再現と同じ手順・同じ乱数で，M0 のソースの顔ベクトルを作る．

    引数:
        context (pipeline.Context): 再現用の画像について準備したモデルとデータ．
    戻り値:
        sources (list[tuple]): (条件の名前, 形状 (n, 512) の顔ベクトル) のリスト．Self は None．
    """
    import common as ex002_common

    ex002_common.NUM_SAMPLES = 1
    set_seed(settings.SEED)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.SEED)
    mask = torch.ones(context.mu.shape[1], device="cuda")
    sources: List[Tuple[str, Optional[torch.Tensor]]] = []
    for alpha in settings.REPRODUCTION_ALPHAS:
        setting = samplers.Setting("M0", "M0", "alpha", alpha, alpha=alpha)
        vectors = samplers.sample(setting, context.vae, context.v, context.v0, context.mu, context.std, mask, generator, None)[:, 0]
        sources.append((f"M0|alpha={alpha:g}", vectors))
    return [("Self", None)] + sources


def main_sources(context: pipeline.Context) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """
    本実験の主要な設定 (Self，Recon，M0 の alpha=2，各方式の再識別率0.5に相当する設定) の
    ソースの顔ベクトルを作る．ノルムの補正は行わない．

    引数:
        context (pipeline.Context): 対象の画像について準備したモデルとデータ．
    戻り値:
        sources (list[tuple]): (条件の名前, 形状 (n, 512) の顔ベクトル) のリスト．Self は None．
    """
    import calibrate
    import common as ex002_common
    import generate
    from attributes import load_ranking

    ex002_common.NUM_SAMPLES = 1
    chosen = calibrate.load_chosen()
    half = chosen[chosen["target_reid_top1"] == 0.5]
    wanted = [("M0", 0.0), ("M0", 2.0)] + [(row.method, row.param) for row in half.itertuples()]
    ranking = load_ranking()
    set_seed(settings.SEED)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.SEED)
    sources: List[Tuple[str, Optional[torch.Tensor]]] = [("Self", None)]
    for method, param in wanted:
        setting = generate.make_setting(method, param)
        mask = samplers.kept_mask(setting, ranking, ranking)
        vectors = samplers.sample(setting, context.vae, context.v, context.v0, context.mu, context.std, mask, generator, None)[:, 0]
        name = "Recon" if (method, param) == ("M0", 0.0) else f"{method}|{setting.param_name}={param:g}"
        sources.append((name, vectors))
    return sources


def other_sources(context: pipeline.Context, eval_data: prepare.EvalData) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """
    下限の参照として，重複グループが異なる他人の実際の顔ベクトルをソースにする条件を作る．

    引数:
        context (pipeline.Context): 対象の画像について準備したモデルとデータ．
        eval_data (prepare.EvalData): 評価用データ一式．
    戻り値:
        sources (list[tuple]): (条件の名前, 形状 (n, 512) の顔ベクトル) のリスト．
    """
    import generate

    others = generate.other_index(eval_data.group[context.index], eval_data.group, np.random.default_rng(settings.SEED))
    return [("Other", torch.from_numpy(eval_data.v[others]).cuda())]


@torch.no_grad()
def evaluate_condition(face_pipeline, anonymizer: FixedVectorAnonymizer, models: ghost.Models,
                       paths: List[str], vectors: Optional[torch.Tensor], originals: Dict[str, np.ndarray],
                       name: str, keep: int = 0) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
    """
    1つの条件について，全画像を既存の `anonymize_image` で匿名化し，3通りの入力で評価する．

    引数:
        face_pipeline (FaceAnonymizerPipeline): 既存のパイプライン．
        anonymizer (FixedVectorAnonymizer): パイプラインに組み込んだ `Anonymizer`．
        models (ghost.Models): モデル一式．
        paths (list[str]): 要素数 n の画像の相対パス．
        vectors (torch.Tensor または None): 形状 (n, 512) のソースの顔ベクトル．None の場合は自分自身．
        originals (dict): 元画像に対する値．"v" (形状 (n, 512))，"crop_age"，"crop_is_female"，
            "full_age"，"full_is_female" (いずれも形状 (n,))，および再識別に用いる "gallery"，
            "gallery_group"，"group" (GPU上のテンソル)．
        name (str): 進捗表示に用いる条件の名前．
        keep (int): 貼り戻した画像を先頭から何枚保持して返すか．
    戻り値:
        metrics (dict): いずれも形状 (n,) の配列．`{mode}_gender_match`，`{mode}_age_diff` (mode は crop，full，
            recrop)，`cos_crop`，`cos_recrop`，`recrop_detected`，`reid_top1_{crop,recrop}`，`reid_top10_{crop,recrop}`．
            顔を検出できなかった画像の recrop の値は NaN．
        kept (list[np.ndarray]): 貼り戻した画像 (先頭 keep 枚)．
    """
    from utils.inference.image_processing import crop_face

    num_images = len(paths)
    metrics = {key: np.full(num_images, np.nan) for key in
               [f"{mode}_gender_match" for mode in MODES] + [f"{mode}_age_diff" for mode in MODES] + ["cos_crop", "cos_recrop"]}
    metrics["recrop_detected"] = np.zeros(num_images)
    original_v = torch.from_numpy(originals["v"]).cuda()
    kept = []
    embeddings = {"crop": torch.zeros_like(original_v), "recrop": torch.zeros_like(original_v)}

    def record(mode: str, i: int, prediction: Dict[str, np.ndarray], reference_prefix: str) -> None:
        """1枚分の性別の一致と年齢の差を記録する (mode: 入力の種類，i: 画像の番号)．"""
        metrics[f"{mode}_gender_match"][i] = float(prediction["is_female"][0] == originals[f"{reference_prefix}_is_female"][i])
        metrics[f"{mode}_age_diff"][i] = prediction["age"][0] - originals[f"{reference_prefix}_age"][i]

    for i, path in enumerate(tqdm(paths, desc=name, leave=False)):
        anonymizer.vector = None if vectors is None else vectors[i]
        face, full = face_pipeline.anonymize_image(os.path.join(settings.REPO_ROOT, path))
        if face is None:
            continue
        if i < keep:
            kept.append(full)

        record("crop", i, ghost.dex_predict(models, face[None]), "crop")
        v_crop = ghost.embed(models, ghost.resize_to_crop(models, face[None]))
        embeddings["crop"][i] = v_crop[0]
        metrics["cos_crop"][i] = float(F.cosine_similarity(v_crop, original_v[i:i + 1]))

        record("full", i, ghost.dex_predict(models, full[None]), "full")
        try:
            recrop = crop_face(full, models.app, models.crop_size)[0]
        except TypeError:
            continue
        metrics["recrop_detected"][i] = 1.0
        record("recrop", i, ghost.dex_predict(models, recrop[None]), "crop")
        v_recrop = ghost.embed(models, recrop[None])
        embeddings["recrop"][i] = v_recrop[0]
        metrics["cos_recrop"][i] = float(F.cosine_similarity(v_recrop, original_v[i:i + 1]))

    # 再識別率: 生成器が出力した顔 (crop) と，貼り戻した画像から切り出し直した顔 (recrop) のそれぞれで計算する
    for mode, vectors_of_mode in embeddings.items():
        reid = reidentification(vectors_of_mode.unsqueeze(1), originals["gallery"], originals["gallery_group"], originals["group"])
        valid = np.ones(num_images) if mode == "crop" else metrics["recrop_detected"]
        for level in ("top1", "top10"):
            values = reid[f"reid_{level}"][:, 0].cpu().numpy().astype(np.float64)
            metrics[f"reid_{level}_{mode}"] = np.where(valid > 0, values, np.nan)
    return metrics, kept


def summarize(name: str, metrics: Dict[str, np.ndarray]) -> dict:
    """
    1つの条件の画像ごとの指標を要約する．

    引数:
        name (str): 条件の名前．
        metrics (dict): `evaluate_condition` の戻り値．
    戻り値:
        row (dict): cos類似度 (2通り)，3通りの入力での性別一致率・年齢のRMSE・年齢の差の絶対値の平均など．
    """
    row = {"condition": name, "num_images": int(len(metrics["cos_crop"])),
           "recrop_detection_rate": float(metrics["recrop_detected"].mean()),
           "cos_crop": float(np.nanmean(metrics["cos_crop"])), "cos_recrop": float(np.nanmean(metrics["cos_recrop"]))}
    for key in ("reid_top1_crop", "reid_top10_crop", "reid_top1_recrop", "reid_top10_recrop"):
        row[key] = float(np.nanmean(metrics[key]))
    for mode in MODES:
        diff = metrics[f"{mode}_age_diff"]
        row[f"{mode}_gender_match"] = float(np.nanmean(metrics[f"{mode}_gender_match"]))
        row[f"{mode}_age_rmse"] = float(np.sqrt(np.nanmean(diff ** 2)))
        row[f"{mode}_age_abs_diff"] = float(np.nanmean(np.abs(diff)))
        row[f"{mode}_age_bias"] = float(np.nanmean(diff))
    return row


def run(target: str, num_images: int = 0) -> None:
    """
    貼り戻した画像での評価を実行し，条件ごとの要約表と画像ごとの指標を保存する．

    引数:
        target (str): "reproduction" (再現用の200枚で M0 の alpha を掃引)，
            "other" (再現用の200枚で，他人の顔ベクトルをソースにする)，
            または "main" (本実験用の画像で主要な設定を評価)．
        num_images (int): 用いる画像数．0 の場合は対象の全画像．
    戻り値:
        なし．
    """
    eval_data = prepare.load()
    index = eval_data.reproduction_index if target in ("reproduction", "other") else eval_data.main_index
    index = index[:num_images] if num_images > 0 else index
    target_dir = settings.condition_dir(f"pasteback_{target}", f"n{len(index)}", settings.SEED)
    if settings.is_finished(target_dir):
        print(pd.read_csv(os.path.join(target_dir, "summary.csv")).round(4).to_string(index=False))
        return

    models = ghost.load_models()
    face_pipeline, anonymizer = build_pipeline(models)
    context = pipeline.build_context(eval_data, index, models)
    paths = [eval_data.paths[i] for i in index]
    full_images = [cv2.imread(os.path.join(settings.REPO_ROOT, path)) for path in paths]
    full = {key: np.concatenate([ghost.dex_predict(models, image[None])[key] for image in full_images])
            for key in ("age", "is_female")}
    originals = {"v": context.v.cpu().numpy(), "crop_age": context.original_dex["age"],
                 "crop_is_female": context.original_dex["is_female"],
                 "full_age": full["age"], "full_is_female": full["is_female"],
                 "gallery": context.gallery, "gallery_group": context.gallery_group, "group": context.group}
    if target == "reproduction":
        sources = reproduction_sources(context)
    elif target == "other":
        sources = other_sources(context, eval_data)
    else:
        sources = main_sources(context)

    rows, per_image = [], {}
    for name, vectors in sources:
        metrics, kept = evaluate_condition(face_pipeline, anonymizer, models, paths, vectors, originals, name, keep=8)
        rows.append(summarize(name, metrics))
        per_image.update({f"{name}::{key}": value for key, value in metrics.items()})
        np.save(os.path.join(target_dir, f"pasted_{name.replace('|', '__')}.npy"), np.array(kept, dtype=object), allow_pickle=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(target_dir, "summary.csv"), index=False)
    np.savez_compressed(os.path.join(target_dir, "per_image.npz"), index=index, **per_image)
    settings.save_json({"seed": settings.SEED, "target": target, "num_images": int(len(index))},
                       os.path.join(target_dir, "meta.json"))
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="貼り戻した画像での評価を実行する")
    parser.add_argument("--target", choices=["reproduction", "main", "other"], default="reproduction")
    parser.add_argument("--num_images", type=int, default=0)
    arguments = parser.parse_args()
    run(arguments.target, arguments.num_images)
