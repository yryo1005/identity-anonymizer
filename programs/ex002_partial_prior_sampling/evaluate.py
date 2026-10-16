"""
実験B: サンプリング方式の比較．

評価用データの各画像について，各設定で K 本の匿名化後の顔ベクトル v_tilde を生成し，
本人との類似度，再識別率，属性の保持，出力の自然さ，多様性を計算する．

    python programs/ex002_partial_prior_sampling/evaluate.py
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import common
import samplers
from attributes import load_ranking
from classifiers import VARIANTS, AttributePredictor, load_predictor
from data import load_dataset
from machine_learning_utils import set_seed
from model import encode_all, load_model

METHOD = "expB_sampling"
HYPERPARAMETER = f"K{common.NUM_SAMPLES}"


@torch.no_grad()
def reidentification(v_tilde: torch.Tensor, gallery: torch.Tensor, gallery_group: torch.Tensor,
                     group: torch.Tensor, chunk_size: int = 8192) -> Dict[str, torch.Tensor]:
    """
    各サンプルに最も近い実ベクトルを探索対象の全画像から探し，元の画像またはその重複グループが
    上位に入るかを判定する．近さはコサイン類似度で測る．

    引数:
        v_tilde (torch.Tensor): 形状 (N, K, 512) の匿名化後の顔ベクトル．
        gallery (torch.Tensor): 形状 (G, 512) の，L2正規化済みの探索対象の顔ベクトル．
        gallery_group (torch.Tensor): 形状 (G,) の探索対象の重複グループの番号．
        group (torch.Tensor): 形状 (N,) の各画像の重複グループの番号．
        chunk_size (int): 1回に類似度行列を計算するサンプル数．
    戻り値:
        result (dict): いずれも形状 (N, K) の，reid_top1，reid_top10 (0/1) と nn_cos (最近傍との cos)．
    """
    num_images, num_samples, _ = v_tilde.shape
    queries = F.normalize(v_tilde.reshape(num_images * num_samples, -1), dim=1)
    targets = group.repeat_interleave(num_samples)
    top1, top10, nn_cos = [], [], []
    for start in range(0, len(queries), chunk_size):
        values, indices = (queries[start:start + chunk_size] @ gallery.T).topk(10, dim=1)
        hits = gallery_group[indices] == targets[start:start + chunk_size].unsqueeze(1)
        top1.append(hits[:, 0].float())
        top10.append(hits.any(dim=1).float())
        nn_cos.append(values[:, 0])
    shape = (num_images, num_samples)
    return {"reid_top1": torch.cat(top1).view(shape), "reid_top10": torch.cat(top10).view(shape),
            "nn_cos": torch.cat(nn_cos).view(shape)}


@torch.no_grad()
def attribute_metrics(v_tilde: torch.Tensor, predictor: AttributePredictor, reference: Dict[str, torch.Tensor],
                      labels: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """
    匿名化後の顔ベクトルの属性の予測を，元の顔ベクトルの予測および正解ラベルと比べる．

    引数:
        v_tilde (torch.Tensor): 形状 (N, K, 512) の匿名化後の顔ベクトル．
        predictor (AttributePredictor): 属性予測器．
        reference (dict): 元の顔ベクトル v に対する予測 (各値は形状 (N,))．
        labels (dict): 正解ラベル (各値は形状 (N,))．
        prefix (str): 指標名の接頭辞 (正規化なしの予測器は ""，L2正規化ありは "l2_")．
    戻り値:
        result (dict): いずれも形状 (N, K) の指標．gender_agree / race_agree (v の予測との一致)，
            age_diff (v の予測との差の絶対値)，gender_acc / race_acc (正解との一致)，age_err (正解との差の絶対値)．
    """
    num_images, num_samples, _ = v_tilde.shape
    prediction = predictor.predict(v_tilde.reshape(num_images * num_samples, -1))
    prediction = {name: value.view(num_images, num_samples) for name, value in prediction.items()}
    result = {}
    for name in ("gender", "race"):
        result[f"{prefix}{name}_agree"] = (prediction[name] == reference[name].unsqueeze(1)).float()
        result[f"{prefix}{name}_acc"] = (prediction[name] == labels[name].unsqueeze(1)).float()
    result[f"{prefix}age_diff"] = (prediction["age"] - reference["age"].unsqueeze(1)).abs().float()
    result[f"{prefix}age_err"] = (prediction["age"] - labels["age"].unsqueeze(1)).abs().float()
    return result


def pairwise_cosine(v_tilde: torch.Tensor) -> torch.Tensor:
    """
    同じ画像から得た K 本のサンプルについて，異なる2本の組のコサイン類似度の平均を計算する．

    引数:
        v_tilde (torch.Tensor): 形状 (N, K, 512) の匿名化後の顔ベクトル．
    戻り値:
        diversity_cos (torch.Tensor): 形状 (N,) の，サンプル同士のコサイン類似度の平均．
    """
    num_samples = v_tilde.shape[1]
    total = F.normalize(v_tilde, dim=2).sum(dim=1)
    return (total.pow(2).sum(dim=1) - num_samples) / (num_samples * (num_samples - 1))


def summarize(per_image: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    画像ごとの指標から，全体の平均と分位点を計算する．

    引数:
        per_image (dict): 指標名をkey，形状 (N,) の配列をvalueとする辞書．
    戻り値:
        summary (dict): `{指標}_mean`，`{指標}_q05`，`{指標}_q95` をkeyとする辞書．
    """
    summary = {}
    for name, values in per_image.items():
        summary[f"{name}_mean"] = float(values.mean())
        for q, value in zip(common.QUANTILES, np.quantile(values, common.QUANTILES)):
            summary[f"{name}_q{int(round(q * 100)):02d}"] = float(value)
    return summary


def other_image_index(num_images: int, seed: int) -> torch.Tensor:
    """
    各画像に対し，自分以外の画像の番号を K 個ずつ割り当てる (参照用の「ランダムな実画像」)．

    引数:
        num_images (int): 画像数 N．
        seed (int): 乱数シード．
    戻り値:
        index (torch.Tensor): 形状 (N, K) の画像番号 (GPU上)．index[i, k] != i を満たす．
    """
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, num_images, size=(num_images, common.NUM_SAMPLES))
    return torch.from_numpy((np.arange(num_images)[:, None] + offsets) % num_images).cuda()


def run_seed(seed: int) -> None:
    """
    1つのseedについて実験Bを実行し，設定ごとの要約表と画像ごとの指標を保存する．

    引数:
        seed (int): 乱数シード．
    戻り値:
        なし．
    """
    target_dir = common.condition_dir(METHOD, HYPERPARAMETER, seed)
    if common.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
        return

    dataset = load_dataset()
    model = load_model()
    evaluation = dataset.is_eval
    v = torch.from_numpy(dataset.v[evaluation]).cuda()
    mu, std = encode_all(model, dataset.v[evaluation])
    with torch.no_grad():
        v0 = model.decode(mu)
    group = torch.from_numpy(dataset.group[evaluation]).cuda()
    gallery = F.normalize(torch.from_numpy(dataset.gallery_v).cuda(), dim=1)
    gallery_group = torch.from_numpy(dataset.gallery_group).cuda()
    labels = {name: torch.from_numpy(getattr(dataset, name)[evaluation]).cuda() for name in ("gender", "race", "age")}
    predictors = {variant: load_predictor(variant) for variant in VARIANTS}
    references = {variant: predictor.predict(v) for variant, predictor in predictors.items()}
    prefixes = {"raw": "", "l2": "l2_"}

    set_seed(seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    ranking = load_ranking()
    random_ranking = np.random.default_rng(seed).permutation(len(ranking))
    other = other_image_index(v.shape[0], seed)

    rows, stacked = [], {}
    settings = samplers.all_settings()
    for setting in tqdm(settings, desc=f"expB seed={seed}", leave=False):
        mask = samplers.kept_mask(setting, ranking, random_ranking) if setting.is_latent else None
        v_tilde = samplers.sample(setting, model, v, v0, mu, std, mask, generator, other)

        per_sample = {
            "cos_v": F.cosine_similarity(v_tilde, v.unsqueeze(1), dim=2),
            "cos_v0": F.cosine_similarity(v_tilde, v0.unsqueeze(1), dim=2),
            "norm": v_tilde.norm(dim=2),
        }
        per_sample.update(reidentification(v_tilde, gallery, gallery_group, group))
        for variant, predictor in predictors.items():
            per_sample.update(attribute_metrics(v_tilde, predictor, references[variant], labels, prefixes[variant]))
        per_image = {name: value.mean(dim=1).cpu().numpy() for name, value in per_sample.items()}
        per_image["diversity_cos"] = pairwise_cosine(v_tilde).cpu().numpy()

        row = {"setting": setting.name, "method": setting.method, "curve": setting.curve,
               "param_name": setting.param_name, "param": setting.param}
        row.update(summarize(per_image))
        row.update(samplers.closed_form(setting, model, v, v0, mu, std, mask))
        rows.append(row)
        for name, values in per_image.items():
            stacked.setdefault(name, []).append(values.astype(np.float32))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(target_dir, "summary.csv"), index=False)
    np.savez_compressed(os.path.join(target_dir, "per_image.npz"), settings=np.array([s.name for s in settings]),
                        **{name: np.stack(values) for name, values in stacked.items()})
    common.save_json({
        "seed": seed, "num_samples": common.NUM_SAMPLES, "num_eval_images": int(v.shape[0]),
        "num_gallery_images": int(gallery.shape[0]), "num_settings": len(settings),
        "attribute_ranking": [int(j) for j in ranking],
        "random_ranking_for_M2r": [int(j) for j in random_ranking],
    }, os.path.join(target_dir, "meta.json"))

    columns = ["setting", "cos_v_mean", "pred_cos_v", "reid_top1_mean", "gender_agree_mean", "race_agree_mean",
               "age_diff_mean", "norm_mean", "nn_cos_mean", "diversity_cos_mean"]
    print(f"\n[seed={seed}]\n" + summary[columns].round(3).to_string(index=False))


def run() -> None:
    """
    `common.SEEDS` の全てのseedについて実験Bを実行する．

    引数:
        なし．
    戻り値:
        なし．
    """
    for seed in common.SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    run()
