"""
GHOSTによる顔の生成と，生成した顔画像の評価 (ArcFaceの再抽出，DEXによる年齢・性別の推定) を
バッチで行うためのモジュール．

処理の内容は，既存の `FaceAnonymizerPipeline.anonymize_image` (生成) と
`identity_anonymizer.evaluation.metrics.evaluate_single_image` (評価) と同一である．
既存の実装は1枚ずつ処理し，生成した顔を元画像へ貼り戻すためにランドマーク検出器 (mxnet) を
読み込むが，評価に使うのは貼り戻す前の顔画像だけであるため，本モジュールは顔検出器，生成器，
ArcFace，DEXのみを読み込む．
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import settings


@dataclass
class Models:
    """
    生成と評価に用いるモデル一式．

    属性:
        app: 顔検出・切り出しモデル (Face_detect_crop)．
        generator (torch.nn.Module): GHOSTの生成器 (半精度，GPU上)．
        arcface (torch.nn.Module): ArcFace (iresnet100，GPU上)．
        dex: DEXの年齢・性別推定モデル (DexModels)．読み込まない場合は None．
        crop_size (int): GHOSTが前提とする顔画像の一辺のサイズ．
    """

    app: object
    generator: Optional[torch.nn.Module]
    arcface: torch.nn.Module
    dex: object
    crop_size: int = 224


def patch_onnxruntime(num_threads: int = 0) -> None:
    """
    insightface==0.2.1 が providers 無しで `onnxruntime.InferenceSession` を生成して例外になるのを
    避けるため，このプロセス内に限り providers の既定値を補う．あわせてスレッド数を制限する．

    引数:
        num_threads (int): 1セッションが使用するCPUスレッド数．0 の場合は制限しない．
    戻り値:
        なし．
    """
    import onnxruntime

    original_session = onnxruntime.InferenceSession
    if getattr(original_session, "_ex003_patched", False):
        return

    class _SessionWithDefaultProviders(original_session):
        _ex003_patched = True

        def __init__(self, path, sess_options=None, providers=None, **kwargs):
            providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if sess_options is None and num_threads > 0:
                sess_options = onnxruntime.SessionOptions()
                sess_options.intra_op_num_threads = num_threads
            super().__init__(path, sess_options, providers=providers, **kwargs)

    onnxruntime.InferenceSession = _SessionWithDefaultProviders


def load_models(with_generator: bool = True, with_dex: bool = True, num_threads: int = 0) -> Models:
    """
    顔検出器，生成器，ArcFace，DEXを読み込む．設定値は既存の `load_ghost_models` と同じである．

    引数:
        with_generator (bool): GHOSTの生成器を読み込むかどうか．
        with_dex (bool): DEXを読み込むかどうか．
        num_threads (int): 顔検出器が使用するCPUスレッド数．0 の場合は制限しない．
    戻り値:
        models (Models): 読み込んだモデル一式．
    """
    if num_threads > 0:
        cv2.setNumThreads(num_threads)
        torch.set_num_threads(num_threads)
    patch_onnxruntime(num_threads)
    from identity_anonymizer import config
    from arcface_model.iresnet import iresnet100
    from insightface_func.face_detect_crop_multi import Face_detect_crop
    from network.AEI_Net import AEI_Net

    weights_dir = config.GHOST_WEIGHTS_DIR
    app = Face_detect_crop(name="antelope", root=weights_dir)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    generator = None
    if with_generator:
        generator = AEI_Net(backbone="unet", num_blocks=2, c_id=512)
        generator.eval()
        generator.load_state_dict(torch.load(os.path.join(weights_dir, "G_unet_2blocks.pth"), map_location="cpu"))
        generator = generator.cuda().half()

    arcface = iresnet100(fp16=False)
    arcface.load_state_dict(torch.load(os.path.join(weights_dir, "backbone.pth"), map_location="cpu"))
    arcface = arcface.cuda().eval()

    dex = None
    if with_dex:
        from identity_anonymizer.evaluation.age_gender import load_dex_models
        dex = load_dex_models(device="cuda")
    return Models(app=app, generator=generator, arcface=arcface, dex=dex)


@torch.no_grad()
def embed(models: Models, faces_bgr: np.ndarray) -> torch.Tensor:
    """
    224x224 の顔画像からArcFaceの顔ベクトルを抽出する (`FaceAnonymizerPipeline.get_embedding` と同じ処理)．

    引数:
        models (Models): モデル一式．
        faces_bgr (np.ndarray): 形状 (B, 224, 224, 3) の顔画像 (BGR，uint8)．
    戻り値:
        embeddings (torch.Tensor): 形状 (B, 512) の顔ベクトル (GPU上，正規化なし)．
    """
    from utils.inference.image_processing import normalize_and_torch_batch

    faces_norm = normalize_and_torch_batch(np.ascontiguousarray(faces_bgr[:, :, :, ::-1]))
    return models.arcface(F.interpolate(faces_norm, scale_factor=0.5, mode="bilinear", align_corners=True))


def crop_target(models: Models, image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    1枚の画像から，匿名化の対象となる顔を切り出す．`FaceAnonymizerPipeline.anonymize_image` と
    同じ関数を同じ順序で呼び出す．

    引数:
        models (Models): モデル一式．
        image_bgr (np.ndarray): 形状 (H, W, 3) の画像 (BGR)．
    戻り値:
        crops (tuple または None): 顔が検出できない場合は None．検出できた場合は次の2つ．
            target224 (np.ndarray): 形状 (224, 224, 3) の顔画像．元画像の顔ベクトルとDEXの入力に用いる．
            resized256 (np.ndarray): 形状 (256, 256, 3) の顔画像．生成器のターゲットに用いる．
    """
    from utils.inference.video_processing import crop_frames_and_get_transforms, get_target, resize_frames

    full_frames = [image_bgr]
    try:
        target = get_target(full_frames, models.app, models.crop_size)
    except TypeError:
        return None
    if target is None:
        return None
    target_embed = embed(models, np.array(target))
    crop_frames_list, _ = crop_frames_and_get_transforms(
        full_frames, target_embed, models.app, models.arcface, models.crop_size, set_target=False, similarity_th=0.15,
    )
    resized, present = resize_frames(crop_frames_list[0])
    if len(resized) == 0 or present[0] == 0:
        return None
    return target[0], resized[0]


@torch.no_grad()
def generate(models: Models, resized256: np.ndarray, sources: torch.Tensor) -> np.ndarray:
    """
    ターゲットの顔画像とソースの顔ベクトルから，GHOSTで顔画像を生成する．
    `transform_target_to_torch` と `faceshifter_batch` と同じ処理を，ターゲットごとに異なる
    ソースを与えられるようバッチ化したものである．

    引数:
        models (Models): モデル一式．
        resized256 (np.ndarray): 形状 (B, 256, 256, 3) のターゲットの顔画像 (BGR，uint8)．
        sources (torch.Tensor): 形状 (B, 512) のソースの顔ベクトル．
    戻り値:
        generated (np.ndarray): 形状 (B, 256, 256, 3) の生成した顔画像 (BGR，uint8)．
    """
    from utils.inference.core import transform_target_to_torch

    target = transform_target_to_torch(resized256, half=True)
    output, _ = models.generator(target, sources.cuda().half())
    output = (output.permute(0, 2, 3, 1) * 0.5 + 0.5) * 255
    return output[:, :, :, [2, 1, 0]].type(torch.uint8).cpu().numpy()


def resize_to_crop(models: Models, faces_bgr: np.ndarray) -> np.ndarray:
    """
    顔画像を 224x224 にリサイズする (生成した 256x256 の顔画像をArcFaceとDEXへ入力する前の処理)．

    引数:
        models (Models): モデル一式．
        faces_bgr (np.ndarray): 形状 (B, H, W, 3) の顔画像．
    戻り値:
        resized (np.ndarray): 形状 (B, 224, 224, 3) の顔画像．
    """
    size = (models.crop_size, models.crop_size)
    return np.stack([cv2.resize(face, size) for face in faces_bgr])


@torch.no_grad()
def dex_predict(models: Models, faces_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    DEXで年齢と性別を推定する．前処理 (224x224 へのリサイズ，BGRのまま，0〜255，正規化なし) と
    出力の定義は `identity_anonymizer.evaluation.age_gender` と同一である．

    引数:
        models (Models): モデル一式．
        faces_bgr (np.ndarray): 形状 (B, H, W, 3) の顔画像 (BGR，uint8)．
    戻り値:
        prediction (dict): "age" (形状 (B,)，見た目年齢の期待値)，"female_prob" と "male_prob" (形状 (B,))，
            "is_female" (形状 (B,)，女性の確率が男性の確率より高ければ True)．
    """
    tensor = torch.from_numpy(np.ascontiguousarray(resize_to_crop(models, faces_bgr).transpose(0, 3, 1, 2))).float().cuda()
    age_probs = models.dex.age_model(tensor).double()
    gender_probs = models.dex.gender_model(tensor).double().cpu().numpy()
    ages = torch.arange(1, age_probs.shape[1] + 1, device="cuda", dtype=torch.float64)
    return {
        "age": (age_probs * ages).sum(dim=1).cpu().numpy(),
        "female_prob": gender_probs[:, 0],
        "male_prob": gender_probs[:, 1],
        "is_female": gender_probs[:, 0] > gender_probs[:, 1],
    }
