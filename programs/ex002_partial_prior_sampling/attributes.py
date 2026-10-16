"""
実験A: 潜在次元ごとの属性関連度．

潜在変数の平均 mu の各次元について，性別・人種・年代に対する相関比 eta^2 と，年齢との
Spearmanの順位相関を計算する．総合スコア r_j は3つの eta^2 の最大値とする．
さらに，r_j の上位 k 次元だけを使った属性分類器の精度を，全次元を使った場合およびランダムに
選んだ k 次元を使った場合と比べ，属性が少数の座標軸に揃っているかを確認する．

    python programs/ex002_partial_prior_sampling/attributes.py
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from data import Dataset, age_bin, load_dataset
from model import encode_all, load_model

METHOD = "expA_attribute_relevance"
HYPERPARAMETER = "mu"
#: ランダムに選んだ k 次元での分類を繰り返す回数
NUM_RANDOM_SUBSETS = 5


def correlation_ratio(x: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """
    各次元について，カテゴリに対する相関比 eta^2 (グループ間の分散 / 全体の分散) を計算する．

    引数:
        x (np.ndarray): 形状 (N, d) の値．
        categories (np.ndarray): 形状 (N,) のカテゴリ番号．
    戻り値:
        eta_squared (np.ndarray): 形状 (d,) の相関比．
    """
    total = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
    between = np.zeros(x.shape[1])
    for category in np.unique(categories):
        members = x[categories == category]
        between += len(members) * (members.mean(axis=0) - x.mean(axis=0)) ** 2
    return between / total


def relevance_table(mu: np.ndarray, dataset_age: np.ndarray, gender: np.ndarray, race: np.ndarray) -> pd.DataFrame:
    """
    次元ごとの属性関連度の表を作る．

    引数:
        mu (np.ndarray): 形状 (N, d) の潜在変数の平均．
        dataset_age (np.ndarray): 形状 (N,) の年齢．
        gender (np.ndarray): 形状 (N,) の性別．
        race (np.ndarray): 形状 (N,) の人種．
    戻り値:
        table (pd.DataFrame): 行数 d の表．列は dim，eta2_gender，eta2_race，eta2_age，
            spearman_age，r_max (3つの eta^2 の最大値)，r_sum (3つの eta^2 の和)，
            rank (r_max の降順の順位，0始まり)．
    """
    table = pd.DataFrame({
        "dim": np.arange(mu.shape[1]),
        "eta2_gender": correlation_ratio(mu, gender),
        "eta2_race": correlation_ratio(mu, race),
        "eta2_age": correlation_ratio(mu, age_bin(dataset_age)),
        "spearman_age": [stats.spearmanr(mu[:, j], dataset_age).statistic for j in range(mu.shape[1])],
    })
    eta_columns = ["eta2_gender", "eta2_race", "eta2_age"]
    table["r_max"] = table[eta_columns].max(axis=1)
    table["r_sum"] = table[eta_columns].sum(axis=1)
    table["rank"] = table["r_max"].rank(ascending=False, method="first").astype(int) - 1
    return table


def top_dims(table: pd.DataFrame, k: int) -> np.ndarray:
    """
    属性関連度 r_max の上位 k 次元の番号を返す．

    引数:
        table (pd.DataFrame): `relevance_table` の戻り値．
        k (int): 選ぶ次元数．
    戻り値:
        dims (np.ndarray): 形状 (k,) の次元の番号 (関連度の降順)．
    """
    return table.sort_values("rank")["dim"].to_numpy()[:k]


def probe_accuracy(features: np.ndarray, dataset: Dataset) -> Dict[str, float]:
    """
    与えた特徴量から属性を予測する線形モデルを学習用データで学習し，評価用データでの精度を求める．

    引数:
        features (np.ndarray): 形状 (N, k) の特徴量．
        dataset (Dataset): ラベルと分割を含むデータ一式．
    戻り値:
        accuracy (dict): gender_acc，race_acc (正解率) と age_mae (平均絶対誤差) の辞書．
    """
    train, test = ~dataset.is_eval, dataset.is_eval
    classifier = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    gender_model = classifier().fit(features[train], dataset.gender[train])
    race_model = classifier().fit(features[train], dataset.race[train])
    regressor.fit(features[train], dataset.age[train])
    return {
        "gender_acc": float(gender_model.score(features[test], dataset.gender[test])),
        "race_acc": float(race_model.score(features[test], dataset.race[test])),
        "age_mae": float(np.abs(regressor.predict(features[test]) - dataset.age[test]).mean()),
    }


def probe_rows(mu: np.ndarray, dataset: Dataset, table: pd.DataFrame) -> List[dict]:
    """
    上位 k 次元と，ランダムに選んだ k 次元のそれぞれで分類精度を求める．

    引数:
        mu (np.ndarray): 形状 (N, d) の潜在変数の平均．
        dataset (Dataset): ラベルと分割を含むデータ一式．
        table (pd.DataFrame): 学習用データで計算した `relevance_table`．
    戻り値:
        rows (list[dict]): (選び方, k) ごとの精度の辞書のリスト．
    """
    rng = np.random.default_rng(common.SPLIT_SEED)
    rows = []
    for k in common.TOP_K_FOR_PROBE:
        rows.append({"selection": "top", "k": k, **probe_accuracy(mu[:, top_dims(table, k)], dataset)})
        if k == mu.shape[1]:
            continue
        random_results = [
            probe_accuracy(mu[:, rng.choice(mu.shape[1], size=k, replace=False)], dataset)
            for _ in range(NUM_RANDOM_SUBSETS)
        ]
        mean = {key: float(np.mean([result[key] for result in random_results])) for key in random_results[0]}
        rows.append({"selection": "random", "k": k, **mean})
    return rows


def run() -> None:
    """
    実験Aを実行し，属性関連度の表，分類精度の表，集計を保存する．

    引数:
        なし．
    戻り値:
        なし．
    """
    target_dir = common.condition_dir(METHOD, HYPERPARAMETER, 0)
    if common.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
        return

    dataset = load_dataset()
    mu = encode_all(load_model(), dataset.v)[0].cpu().numpy().astype(np.float64)
    train = ~dataset.is_eval

    table_all = relevance_table(mu, dataset.age, dataset.gender, dataset.race)
    table_train = relevance_table(mu[train], dataset.age[train], dataset.gender[train], dataset.race[train])
    probes = pd.DataFrame(probe_rows(mu, dataset, table_train))
    chance = {
        "gender_acc": float(np.bincount(dataset.gender[train]).max() / train.sum()),
        "race_acc": float(np.bincount(dataset.race[train]).max() / train.sum()),
        "age_mae": float(np.abs(dataset.age[dataset.is_eval] - np.median(dataset.age[train])).mean()),
    }

    sorted_r = np.sort(table_all["r_max"].to_numpy())[::-1]
    cumulative = np.cumsum(sorted_r) / sorted_r.sum()
    summary = {
        "num_images": int(len(mu)),
        "latent_dim": int(mu.shape[1]),
        "r_max_max": float(sorted_r[0]), "r_max_median": float(np.median(sorted_r)), "r_max_min": float(sorted_r[-1]),
        "cumulative_share_top": {str(k): float(cumulative[k - 1]) for k in (8, 16, 32, 64)},
        "uniform_share_top": {str(k): k / mu.shape[1] for k in (8, 16, 32, 64)},
        "eta2_max": {column: float(table_all[column].max()) for column in ("eta2_gender", "eta2_race", "eta2_age")},
        "abs_spearman_age_max": float(table_all["spearman_age"].abs().max()),
        "rank_spearman_all_vs_train": float(stats.spearmanr(table_all["r_max"], table_train["r_max"]).statistic),
        "top32_overlap_all_vs_train": int(len(set(top_dims(table_all, 32)) & set(top_dims(table_train, 32)))),
        "majority_baseline": chance,
    }

    table_all.to_csv(os.path.join(target_dir, "relevance_all_images.csv"), index=False)
    table_train.to_csv(os.path.join(target_dir, "relevance_train_split.csv"), index=False)
    probes.to_csv(os.path.join(target_dir, "probe_accuracy.csv"), index=False)
    common.save_json(summary, os.path.join(target_dir, "summary.json"))
    common.save_json({"num_random_subsets": NUM_RANDOM_SUBSETS, "top_k": common.TOP_K_FOR_PROBE},
                     os.path.join(target_dir, "meta.json"))
    print(pd.Series({k: v for k, v in summary.items()}).to_string())
    print(probes.round(4).to_string(index=False))


def load_ranking() -> np.ndarray:
    """
    学習用データで計算した属性関連度の降順に並べた，次元の番号を読み込む．

    引数:
        なし．
    戻り値:
        ranking (np.ndarray): 形状 (d,) の次元の番号 (関連度の降順)．
    """
    path = os.path.join(common.OUTPUT_DIR, METHOD, HYPERPARAMETER, "0", "relevance_train_split.csv")
    table = pd.read_csv(path)
    return top_dims(table, len(table))


if __name__ == "__main__":
    run()
