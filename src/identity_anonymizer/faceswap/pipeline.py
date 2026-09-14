import os
import shutil
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.faceswap.models import GhostModels

# third_party/ghost 側のユーティリティ(identity_anonymizer.config が sys.path を通す)
from utils.inference.image_processing import get_final_image, normalize_and_torch_batch
from utils.inference.video_processing import (
    read_video,
    get_target,
    get_final_video,
    add_audio_from_another_video,
    face_enhancement,
    crop_frames_and_get_transforms,
    resize_frames,
)
from utils.inference.core import transform_target_to_torch
from utils.inference.faceshifter_run import faceshifter_batch


class FaceAnonymizerPipeline:
    """
    GHOSTによる顔交換と，差し替え可能な `Anonymizer` を組み合わせて，画像・動画中の人物の顔を
    属性を保持したまま匿名化するパイプライン．

    `Anonymizer` の実装(`VAEAnonymizer` 等)はコンストラクタで注入するため，将来的に別の
    匿名化モデルへ差し替える場合は，`anonymizer` 引数に新しいモデルのインスタンスを渡すだけでよく，
    本クラス自体の変更は不要である．
    """

    def __init__(self, models: GhostModels, anonymizer: Anonymizer):
        """
        引数:
            models (GhostModels): `identity_anonymizer.faceswap.models.load_ghost_models` で
                読み込んだGHOST関連のモデル一式．
            anonymizer (Anonymizer): 顔ベクトルを匿名化するモデル．
        戻り値:
            なし．
        """
        self.models = models
        self.anonymizer = anonymizer.cuda().eval()
        self.crop_size = models.crop_size

    def get_embedding(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        顔画像1枚からArcFaceの顔ベクトルを計算する．

        引数:
            image_bgr (np.ndarray): 形状 (224, 224, 3) の切り出し済み顔画像(BGR)．
        戻り値:
            embedding (torch.Tensor): 形状 (1, 512) の顔ベクトル．
        """
        image_norm = normalize_and_torch_batch(np.array([image_bgr[:, :, ::-1]]))
        with torch.no_grad():
            embedding = self.models.arcface(
                F.interpolate(image_norm, scale_factor=0.5, mode="bilinear", align_corners=True)
            )
        return embedding

    def anonymize_image(
        self,
        target_image_path: str,
        out_image_path: Optional[str] = None,
        anonymizer_input: Optional[Any] = None,
        **anonymizer_kwargs,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        1枚の画像に対し，`Anonymizer` で匿名化した顔ベクトルを用いて自分自身の顔に変換する．

        引数:
            target_image_path (str): 匿名化対象の画像のパス．
            out_image_path (str または None): 出力画像の保存先パス．None の場合は保存しない．
            anonymizer_input (Any または None): `Anonymizer.anonymize` に渡す入力．None の場合は
                ターゲット画像から計算したArcFace顔ベクトル(`target_embed`)を用いる
                (`VAEAnonymizer` 等，顔ベクトルを入力とするモデル向けの既定動作)．
                `AttributeNNAnonymizer` のように顔ベクトル以外を入力とするモデルを使う場合は，
                ここに直接入力(例: `{"age": 20, "gender": 0, "race": 0}`)を指定する．
                ターゲット画像自体は，この場合も向き・表情の情報源として使われる．
            **anonymizer_kwargs: `Anonymizer.anonymize` に渡す追加のキーワード引数
                (例: `VAEAnonymizer` の `noise_level`)．
        戻り値:
            face_image (np.ndarray または None): 形状 (256, 256, 3) の変換後の顔画像．
            full_image (np.ndarray または None): 元画像と同じ形状の，顔部分のみ貼り替えた画像．
                顔が検出できない場合は (None, None)．
        """
        target_full = cv2.imread(target_image_path)
        full_frames = [target_full]

        try:
            target = get_target(full_frames, self.models.app, self.crop_size)
        except TypeError:
            target = None
        if target is None:
            print("顔を検出できませんでした．別の画像を使用してください．")
            return None, None

        target_norm = normalize_and_torch_batch(np.array(target))
        target_embed = self.models.arcface(
            F.interpolate(target_norm, scale_factor=0.5, mode="bilinear", align_corners=True)
        )
        crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(
            full_frames, target_embed, self.models.app, self.models.arcface, self.crop_size,
            set_target=False, similarity_th=0.15,
        )

        anonymizer_x = target_embed if anonymizer_input is None else anonymizer_input
        with torch.no_grad():
            source_embed = self.anonymizer.anonymize(anonymizer_x, **anonymizer_kwargs)

        resized_frs, present = resize_frames(crop_frames_list[0])
        resized_frs = np.array(resized_frs)
        target_batch_rs = transform_target_to_torch(resized_frs, half=True)

        y_st = faceshifter_batch(source_embed.half(), target_batch_rs, self.models.generator)

        full_image = get_final_image([y_st], crop_frames_list, full_frames[0], tfm_array_list, self.models.handler)

        if out_image_path is not None:
            os.makedirs(os.path.dirname(out_image_path) or ".", exist_ok=True)
            cv2.imwrite(out_image_path, full_image)

        return y_st[0], full_image

    def get_unified_identity_embedding(
        self,
        target_video_path: str,
        anonymizer_input: Optional[Any] = None,
        **anonymizer_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, list, float]:
        """
        動画1本に対して，全フレームで共通して使用する匿名化後の顔ベクトルを1つだけ計算する．

        動画中で最初に顔検出に成功したフレームを基準として選び，そのフレームのArcFace顔ベクトル
        を計算した後，`Anonymizer` に1回だけ入力してサンプリングする．ここで得られるベクトルを
        動画の全フレームに使い回すことで，フレームごとに異なる人物へ変換されることを防ぐ．

        引数:
            target_video_path (str): 対象動画のパス．
            anonymizer_input (Any または None): `Anonymizer.anonymize` に渡す入力．None の場合は
                基準フレームから計算したArcFace顔ベクトル(`target_embed`)を用いる．
                `AttributeNNAnonymizer` のように顔ベクトル以外を入力とするモデルを使う場合は，
                ここに直接入力(例: `{"age": 20, "gender": 0, "race": 0}`)を指定する．
            **anonymizer_kwargs: `Anonymizer.anonymize` に渡す追加のキーワード引数
                (例: `VAEAnonymizer` の `noise_level`)．
        戻り値:
            source_embed (torch.Tensor): 形状 (1, 512) の，動画全体で共通の匿名化後の顔ベクトル．
            target_embed (torch.Tensor): 形状 (1, 512) の，基準フレームの元の顔ベクトル．
            full_frames (list[np.ndarray]): 動画の全フレーム(BGR画像)のリスト．
            fps (float): 動画のフレームレート．
        """
        full_frames, fps = read_video(target_video_path)
        target = get_target(full_frames, self.models.app, self.crop_size)
        if target is None:
            raise ValueError(f"動画中に顔を検出できませんでした: {target_video_path}")

        target_norm = normalize_and_torch_batch(np.array(target))
        target_embed = self.models.arcface(
            F.interpolate(target_norm, scale_factor=0.5, mode="bilinear", align_corners=True)
        )

        anonymizer_x = target_embed if anonymizer_input is None else anonymizer_input
        with torch.no_grad():
            source_embed = self.anonymizer.anonymize(anonymizer_x, **anonymizer_kwargs)

        return source_embed, target_embed, full_frames, fps

    def anonymize_video(
        self,
        target_video_path: str,
        out_video_path: str,
        similarity_th: float = 0.15,
        batch_size: int = 64,
        use_sr: bool = False,
        keep_audio: bool = True,
        anonymizer_input: Optional[Any] = None,
        **anonymizer_kwargs,
    ) -> str:
        """
        動画1本の中に映る人物の顔を，属性を保持したまま匿名化した架空の顔に変換する．

        動画全体で単一の匿名化後の顔ベクトルを使用するため，出力動画内で人物の見た目は
        時間的に一貫する(同一の架空の人物として現れる)．

        引数:
            target_video_path (str): 匿名化対象の動画のパス．
            out_video_path (str): 出力動画の保存先パス(mp4)．
            similarity_th (float): フレーム内で複数人の顔が検出された場合に，対象人物と
                同一人物とみなす類似度の閾値．
            batch_size (int): GHOST生成器に一度に入力するフレーム数．
            use_sr (bool): 顔領域に超解像を適用するかどうか(`load_ghost_models(use_sr=True)` で
                読み込んだ場合のみ有効)．
            keep_audio (bool): 元動画の音声を出力動画にも付与するかどうか．`ffmpeg` が
                利用できない環境では自動的に無視される．
            anonymizer_input (Any または None): `Anonymizer.anonymize` に渡す入力．
                `get_unified_identity_embedding` を参照．
            **anonymizer_kwargs: `Anonymizer.anonymize` に渡す追加のキーワード引数
                (例: `VAEAnonymizer` の `noise_level`)．
        戻り値:
            out_video_path (str): 出力動画の保存先パス．
        """
        source_embed, target_embed, full_frames, fps = self.get_unified_identity_embedding(
            target_video_path, anonymizer_input=anonymizer_input, **anonymizer_kwargs
        )

        crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(
            full_frames, target_embed, self.models.app, self.models.arcface, self.crop_size,
            set_target=False, similarity_th=similarity_th,
        )

        resized_frs, present = resize_frames(crop_frames_list[0])
        resized_frs = np.array(resized_frs)
        target_batch_rs = transform_target_to_torch(resized_frs, half=True)

        source_embed_half = source_embed.half()
        model_output = []
        for i in tqdm(range(0, target_batch_rs.shape[0], batch_size), leave=False, desc="anonymize_video"):
            # 全バッチで同一の source_embed_half を使い回すことで，動画全体の顔ベクトルを統一する
            y_st = faceshifter_batch(source_embed_half, target_batch_rs[i:i + batch_size], self.models.generator)
            model_output.append(y_st)
        torch.cuda.empty_cache()
        model_output = np.concatenate(model_output)

        final_frames = []
        idx_fs = 0
        for pres in present:
            if pres == 1:
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])
        final_frames_list = [final_frames]

        if use_sr:
            if self.models.super_resolution is None:
                raise ValueError("use_sr=True ですが，load_ghost_models(use_sr=True) で超解像モデルが読み込まれていません．")
            final_frames_list = face_enhancement(final_frames_list, self.models.super_resolution)

        os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
        get_final_video(
            final_frames_list, crop_frames_list, full_frames, tfm_array_list,
            out_video_path, fps, self.models.handler,
        )

        if keep_audio:
            if shutil.which("ffmpeg") is not None:
                add_audio_from_another_video(target_video_path, out_video_path, "audio")
            else:
                print("ffmpeg が見つからないため，音声の付与をスキップしました．")

        return out_video_path
