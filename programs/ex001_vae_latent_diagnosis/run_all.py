"""
ex001_vae_latent_diagnosis の全ての処理 (ArcFaceのキャッシュ，実験1〜3) を順に実行する．
正常終了済みの条件 (`meta.json` が存在する条件) は自動的にスキップされる．

    source programs/ex001_vae_latent_diagnosis/env.sh
    $PYTHON programs/ex001_vae_latent_diagnosis/run_all.py
"""

import argparse

import analysis_required_alpha
import data
import ex1_latent_stats
import ex2_displacement
import ex3_radius_sampling


def main(num_workers: int) -> None:
    """
    ArcFaceのキャッシュの作成，実験1〜3，補足の解析を順に実行する．

    引数:
        num_workers (int): ArcFaceの抽出を並列に実行するプロセス数．
    戻り値:
        なし．
    """
    data.build_arcface_cache(num_workers=num_workers)
    ex1_latent_stats.run()
    ex2_displacement.run()
    ex3_radius_sampling.run()
    analysis_required_alpha.run()
    analysis_required_alpha.within_image_spread()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE潜在空間のサンプリング挙動の診断を一括で実行する")
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args().num_workers)
