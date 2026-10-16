"""
ex002_partial_prior_sampling の全ての処理を順に実行する．
正常終了済みの条件 (`meta.json` が存在する条件) は自動的にスキップされる．
ArcFaceの顔ベクトルは ex001_vae_latent_diagnosis のキャッシュを用いるため，先に ex001 を実行しておくこと．

    source programs/ex002_partial_prior_sampling/env.sh
    $PYTHON programs/ex002_partial_prior_sampling/run_all.py
"""

import analysis
import attributes
import classifiers
import data
import evaluate


def main() -> None:
    """
    ラベルと分割の作成，実験A，属性分類器の学習，実験B，集計を順に実行する．

    引数:
        なし．
    戻り値:
        なし．
    """
    data.build_dataset()
    attributes.run()
    classifiers.run()
    evaluate.run()
    analysis.run()


if __name__ == "__main__":
    main()
