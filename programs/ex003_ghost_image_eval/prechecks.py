"""
作業前の確認．

1. 予稿の再現: 評価用データから無作為に選んだ200枚について，現行方式 M0 で顔画像を生成し，
   cos類似度，DEXの性別一致率，年齢のRMSEを予稿の値と比べる．
2. 線形分類器の健全性: ex002 の属性分類器について，正則化の強さ，標準化の有無，収束を確認する．
3. DEXの精度: UTKFaceの元画像に対するDEXの性別正解率と年齢の誤差を求め，線形分類器と比べる．

    python programs/ex003_ghost_image_eval/prechecks.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import ghost
import pipeline
import prepare
import samplers
import settings
from machine_learning_utils import set_seed

REGULARIZATION_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RIDGE_GRID = [1e-2, 1.0, 1e2, 1e3, 1e4, 1e5]
MAX_ITER = 3000


def reproduce_paper(eval_data: prepare.EvalData, models: ghost.Models) -> pd.DataFrame:
    """
    予稿と同じ条件 (現行方式 M0) で顔画像を生成して評価し，予稿の値と比べる．

    引数:
        eval_data (prepare.EvalData): 評価用データ一式．
        models (ghost.Models): モデル一式．
    戻り値:
        table (pd.DataFrame): alpha ごとに1行の表．実測値，予稿の値，その差を含む．
    """
    import common as ex002_common

    ex002_common.NUM_SAMPLES = 1
    context = pipeline.build_context(eval_data, eval_data.reproduction_index, models)
    set_seed(settings.SEED)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.SEED)
    mask = torch.ones(context.mu.shape[1], device="cuda")

    rows = []
    for alpha in settings.REPRODUCTION_ALPHAS:
        setting = samplers.Setting("M0", "M0", "alpha", alpha, alpha=alpha)
        sources = samplers.sample(setting, context.vae, context.v, context.v0, context.mu, context.std, mask, generator, None)[:, 0]
        metrics, _, _ = pipeline.generate_and_evaluate(context, sources, f"reproduce alpha={alpha}")
        summary = pipeline.summarize(metrics)
        paper_cos, paper_gender, paper_rmse = settings.PAPER_VALUES[alpha]
        rows.append({
            "alpha": alpha, "num_images": summary["num_images"],
            "cos_vector_space": float(torch.nn.functional.cosine_similarity(sources, context.v, dim=1).mean()),
            "cos_measured": summary["cos_gen_v"], "cos_paper": paper_cos, "cos_diff": summary["cos_gen_v"] - paper_cos,
            "gender_match_measured": summary["dex_gender_match"], "gender_match_paper": paper_gender,
            "age_rmse_measured": summary["dex_age_rmse"], "age_rmse_paper": paper_rmse,
            "reid_top1": summary["reid_top1"],
            "within_tolerance": bool(abs(summary["cos_gen_v"] - paper_cos) < settings.REPRODUCTION_TOLERANCE),
        })
    return pd.DataFrame(rows)


#: `notebooks/05_evaluate.ipynb` に記録されている，既存のパイプラインによる評価結果 (サンプル画像467枚)．
#: alpha: (cos類似度, 性別一致率, 年齢RMSE)
REPO_NOTEBOOK_VALUES = {
    0.0: (0.7325, 0.8458, 7.3288),
    1.0: (0.6715, 0.7966, 7.7029),
    1.5: (0.6098, 0.7666, 7.9152),
    2.0: (0.5455, 0.7430, 8.2165),
}


@torch.no_grad()
def reproduce_repo_notebook(models: ghost.Models) -> pd.DataFrame:
    """
    リポジトリ同梱のサンプル画像 (`data/utkface_sample`，500枚) について，現行方式 M0 で生成と評価を行い，
    `notebooks/05_evaluate.ipynb` に記録されている既存のパイプラインの結果と比べる．
    同じ画像に対する比較であるため，本実験の生成・評価の処理が既存のパイプラインと同じかどうかを直接確認できる．

    引数:
        models (ghost.Models): モデル一式．
    戻り値:
        table (pd.DataFrame): alpha ごとに1行の表．実測値，notebookの記録値，予稿の値を含む．
    """
    import cv2
    from model import load_model

    image_dir = os.path.join(settings.REPO_ROOT, "data", "utkface_sample")
    names = sorted(name for name in os.listdir(image_dir) if name.lower().endswith(".jpg"))
    crops = [ghost.crop_target(models, cv2.imread(os.path.join(image_dir, name))) for name in names]
    crops = [crop for crop in crops if crop is not None]
    target224 = np.stack([crop[0] for crop in crops])
    resized256 = np.stack([crop[1] for crop in crops])

    vae = load_model()
    v = torch.cat([ghost.embed(models, target224[i:i + settings.BATCH_SIZE]) for i in range(0, len(crops), settings.BATCH_SIZE)])
    mu, logvar = vae.encode(v)
    std = torch.exp(0.5 * logvar)
    original = pipeline.batched_dex(models, target224)
    set_seed(settings.SEED)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.SEED)

    rows = []
    for alpha in settings.REPRODUCTION_ALPHAS:
        sources = vae.decode(mu + alpha * std * torch.randn(mu.shape, generator=generator, device="cuda"))
        cosines, ages, females = [], [], []
        for start in range(0, len(crops), settings.BATCH_SIZE):
            stop = start + settings.BATCH_SIZE
            generated = ghost.generate(models, resized256[start:stop], sources[start:stop])
            v_gen = ghost.embed(models, ghost.resize_to_crop(models, generated))
            cosines.append(torch.nn.functional.cosine_similarity(v_gen, v[start:stop], dim=1).cpu().numpy())
            dex = ghost.dex_predict(models, generated)
            ages.append(dex["age"])
            females.append(dex["is_female"])
        age_diff = np.concatenate(ages) - original["age"]
        notebook_cos, notebook_gender, notebook_rmse = REPO_NOTEBOOK_VALUES[alpha]
        paper_cos, paper_gender, paper_rmse = settings.PAPER_VALUES[alpha]
        rows.append({
            "alpha": alpha, "num_images": len(crops), "num_files": len(names),
            "cos_measured": float(np.concatenate(cosines).mean()), "cos_notebook": notebook_cos, "cos_paper": paper_cos,
            "gender_match_measured": float((np.concatenate(females) == original["is_female"]).mean()),
            "gender_match_notebook": notebook_gender, "gender_match_paper": paper_gender,
            "age_rmse_measured": float(np.sqrt((age_diff ** 2).mean())),
            "age_rmse_notebook": notebook_rmse, "age_rmse_paper": paper_rmse,
        })
    return pd.DataFrame(rows)


def fit_with_convergence(model, x: np.ndarray, y: np.ndarray):
    """
    モデルを学習し，収束の警告が出たかどうかを記録する．

    引数:
        model: scikit-learn の推定器．
        x (np.ndarray): 形状 (N, D) の特徴量．
        y (np.ndarray): 形状 (N,) の目的変数．
    戻り値:
        model: 学習済みの推定器．
        warned (bool): 収束の警告が出た場合は True．
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return model, any(issubclass(w.category, ConvergenceWarning) for w in caught)


def classifier_sanity() -> pd.DataFrame:
    """
    ex002 と同じデータ分割で，線形の属性分類器の正則化の強さと標準化の有無を変えて学習し直し，
    評価用データでの精度，収束の警告の有無，反復回数を記録する．

    引数:
        なし．
    戻り値:
        table (pd.DataFrame): (属性, 標準化の有無, 正則化の強さ) ごとに1行の表．
    """
    from data import load_dataset

    dataset = load_dataset()
    train, test = ~dataset.is_eval, dataset.is_eval
    raw = dataset.v.astype(np.float64)
    features = {"standardized": StandardScaler().fit(raw[train]).transform(raw), "raw": raw}

    rows = []
    for (scaling, x), c in [(item, c) for item in features.items() for c in REGULARIZATION_GRID]:
        for task, y in (("gender", dataset.gender), ("race", dataset.race)):
            model, warned = fit_with_convergence(LogisticRegression(C=c, max_iter=MAX_ITER), x[train], y[train])
            rows.append({
                "task": task, "scaling": scaling, "regularization": f"C={c:g}",
                "train_score": float(model.score(x[train], y[train])), "eval_score": float(model.score(x[test], y[test])),
                "n_iter": int(np.max(model.n_iter_)), "convergence_warning": warned,
                "weight_norm": float(np.linalg.norm(model.coef_ - model.coef_.mean(axis=0, keepdims=True))
                                     if task == "race" else np.linalg.norm(model.coef_)),
            })
    for (scaling, x), alpha in [(item, a) for item in features.items() for a in RIDGE_GRID]:
        model = Ridge(alpha=alpha).fit(x[train], dataset.age[train])
        rows.append({
            "task": "age", "scaling": scaling, "regularization": f"alpha={alpha:g}",
            "train_score": float(np.abs(model.predict(x[train]) - dataset.age[train]).mean()),
            "eval_score": float(np.abs(model.predict(x[test]) - dataset.age[test]).mean()),
            "n_iter": 0, "convergence_warning": False, "weight_norm": float(np.linalg.norm(model.coef_)),
        })
    return pd.DataFrame(rows)


def dex_accuracy(eval_data: prepare.EvalData, models: ghost.Models) -> dict:
    """
    UTKFaceの元画像 (切り出した顔画像) に対するDEXの精度を，評価用データの全画像と本実験の1,000枚に
    ついて求め，同じ画像に対する線形分類器の精度と並べる．

    引数:
        eval_data (prepare.EvalData): 評価用データ一式．
        models (ghost.Models): モデル一式．
    戻り値:
        result (dict): 画像の集合ごとの，DEXと線形分類器の性別正解率・年齢のMAE．
    """
    from classifiers import load_predictor

    dex = pipeline.batched_dex(models, eval_data.target224)
    linear = {name: value.cpu().numpy() for name, value in
              load_predictor("raw").predict(torch.from_numpy(eval_data.v).cuda()).items()}
    subsets = {"all_eval": np.arange(len(eval_data.v)), "main_1000": eval_data.main_index,
               "reproduction_200": eval_data.reproduction_index}
    result = {}
    for name, index in subsets.items():
        result[name] = {
            "num_images": int(len(index)),
            "dex_gender_acc": float((dex["is_female"][index] == (eval_data.gender[index] == 1)).mean()),
            "dex_age_mae": float(np.abs(dex["age"][index] - eval_data.age[index]).mean()),
            "dex_female_rate": float(dex["is_female"][index].mean()),
            "label_female_rate": float((eval_data.gender[index] == 1).mean()),
            "linear_gender_acc": float((linear["gender"][index] == eval_data.gender[index]).mean()),
            "linear_race_acc": float((linear["race"][index] == eval_data.race[index]).mean()),
            "linear_age_mae": float(np.abs(linear["age"][index] - eval_data.age[index]).mean()),
            "dex_linear_gender_agreement": float((dex["is_female"][index] == (linear["gender"][index] == 1)).mean()),
        }
    return result


def run(skip_classifier_sanity: bool = False) -> None:
    """
    作業前の確認を実行し，結果を保存する．

    引数:
        skip_classifier_sanity (bool): True の場合，時間のかかる線形分類器の再学習を省く．
    戻り値:
        なし．
    """
    target_dir = settings.condition_dir("prechecks", "default", settings.SEED)
    eval_data = prepare.load()
    pd.set_option("display.width", 250)

    reproduction_path = os.path.join(target_dir, "paper_reproduction.csv")
    if not os.path.exists(reproduction_path):
        models = ghost.load_models()
        reproduce_paper(eval_data, models).to_csv(reproduction_path, index=False)
        settings.save_json(dex_accuracy(eval_data, models), os.path.join(target_dir, "dex_accuracy.json"))
    repo_path = os.path.join(target_dir, "repo_notebook_reproduction.csv")
    if not os.path.exists(repo_path):
        reproduce_repo_notebook(ghost.load_models()).to_csv(repo_path, index=False)
    print(pd.read_csv(repo_path).round(4).to_string(index=False))
    print(pd.read_csv(reproduction_path).round(4).to_string(index=False))
    print(pd.DataFrame(settings.load_json(os.path.join(target_dir, "dex_accuracy.json"))).round(4).to_string())

    sanity_path = os.path.join(target_dir, "classifier_sanity.csv")
    if not skip_classifier_sanity and not os.path.exists(sanity_path):
        classifier_sanity().to_csv(sanity_path, index=False)
    if os.path.exists(sanity_path):
        print(pd.read_csv(sanity_path).round(4).to_string(index=False))
        settings.save_json({"seed": settings.SEED}, os.path.join(target_dir, "meta.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="作業前の確認を実行する")
    parser.add_argument("--skip_classifier_sanity", action="store_true")
    run(skip_classifier_sanity=parser.parse_args().skip_classifier_sanity)
