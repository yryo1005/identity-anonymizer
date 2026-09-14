from dataclasses import dataclass
from typing import Optional

import cv2
import torch.nn.functional as F

from identity_anonymizer.evaluation.age_gender import DexModels, estimate_age, estimate_gender
from identity_anonymizer.faceswap.pipeline import FaceAnonymizerPipeline
from utils.inference.image_processing import crop_face


@dataclass
class AnonymizationEvalResult:
    """1枚の画像に対する匿名化前後の評価結果．"""

    image_path: str
    original_age: float
    anonymized_age: float
    original_gender: str
    anonymized_gender: str
    cosine_similarity: float


def evaluate_single_image(
    image_path: str,
    pipeline: FaceAnonymizerPipeline,
    dex_models: DexModels,
    noise_level: float = 1.0,
) -> Optional[AnonymizationEvalResult]:
    """
    1枚の画像に対し，匿名化前後の年齢・性別の一致度とArcFaceのコサイン類似度を計算する．

    `004_GHOST_VAE_inference_v2.ipynb` の評価ループと異なり，`output.jpg` のような共有の
    一時ファイルへ書き出さず，すべてメモリ上の配列のまま処理する．これにより，本関数は
    複数プロセスから安全に並列実行できる(`identity_anonymizer.evaluation.parallel` から利用)．

    引数:
        image_path (str): 評価対象の画像のパス．
        pipeline (FaceAnonymizerPipeline): 匿名化パイプライン．
        dex_models (DexModels): 年齢・性別推定モデル．
        noise_level (float): `Anonymizer.anonymize` に渡すスケール係数．
    戻り値:
        result (AnonymizationEvalResult または None): 評価結果．顔が検出できない場合は None．
    """
    original_face_bgr = cv2.imread(image_path)
    if original_face_bgr is None:
        return None

    anonymized_face_bgr, _ = pipeline.anonymize_image(image_path, noise_level=noise_level)
    if anonymized_face_bgr is None:
        return None

    # get_embedding は224x224に切り出し済みの顔画像を想定しているため，元画像も同様に切り出す
    try:
        cropped_original = crop_face(original_face_bgr, pipeline.models.app, pipeline.crop_size)[0]
    except TypeError:
        return None
    original_embedding = pipeline.get_embedding(cropped_original)
    anonymized_embedding = pipeline.get_embedding(cv2.resize(anonymized_face_bgr, (pipeline.crop_size, pipeline.crop_size)))

    cosine_similarity = F.cosine_similarity(original_embedding, anonymized_embedding).item()

    original_age = estimate_age(cropped_original, dex_models)
    female, male = estimate_gender(cropped_original, dex_models)
    original_gender = "Female" if female > male else "Male"

    anonymized_age = estimate_age(anonymized_face_bgr, dex_models)
    female, male = estimate_gender(anonymized_face_bgr, dex_models)
    anonymized_gender = "Female" if female > male else "Male"

    return AnonymizationEvalResult(
        image_path=image_path,
        original_age=original_age,
        anonymized_age=anonymized_age,
        original_gender=original_gender,
        anonymized_gender=anonymized_gender,
        cosine_similarity=cosine_similarity,
    )
