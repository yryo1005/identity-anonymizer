"""
動画に映る全員の顔を匿名化する処理．

処理の流れ:
    1. ffmpeg で動画を正規化する (一定の fps，高さ・長さ・fps の上限，H.264)．
    2. 全フレームで全ての顔を検出する (顔検出はCPUで動くため，複数のスレッドで並列に実行する)．
    3. 顔を切り出し，ArcFaceの顔ベクトルを計算して，人物ごとに追跡する．
    4. 人物ごとに1つ，選んだ方式で匿名化後の顔ベクトルを作る (動画全体で同じ架空の人物になる)．
    5. GHOSTで全ての顔を生成し，元のフレームへ貼り戻す．
    6. 元の動画の音声を付けて，ブラウザで再生できる H.264 の mp4 に変換する．

顔の切り出し，ランドマークの平滑化，貼り戻しは，GHOSTの `crop_frames_and_get_transforms`，
`smooth_landmarks`，`get_final_video` と同じ関数・同じ手順を，複数の人物に対して適用したものである．
"""

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import demo_settings
import methods
import tracking
import ghost
from model import load_model

#: 進捗の報告に用いる処理段ごとの重み (合計 1.0)
STAGE_WEIGHTS = {"normalize": 0.03, "detect": 0.40, "track": 0.07, "generate": 0.15, "paste": 0.30, "encode": 0.05}

ProgressCallback = Callable[[str, float], None]


@dataclass
class FaceTrack:
    """
    1人の人物について，フレームごとの情報を保持する．F はフレーム数である．

    属性:
        kps (list): 要素数 F．その人物の5点のランドマーク (形状 (5, 2))．映っていないフレームは []．
        crops (list): 要素数 F．切り出した顔画像 (224x224)．映っていないフレームは []．
        transforms (list): 要素数 F．切り出しのアフィン変換行列 (形状 (2, 3))．映っていないフレームは []．
        generated (list): 要素数 F．生成した顔画像 (256x256)．映っていないフレームは []．
    """

    kps: list = field(default_factory=list)
    crops: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    generated: list = field(default_factory=list)


def run_command(command: List[str]) -> None:
    """
    外部コマンドを実行し，失敗した場合は標準エラー出力を含めて例外を送出する．

    引数:
        command (list[str]): 実行するコマンドと引数．
    戻り値:
        なし．
    """
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} の実行に失敗しました: {result.stderr[-800:]}")


def probe(path: str) -> dict:
    """
    ffprobe で動画の情報 (映像の fps，解像度，長さ，音声の有無) を調べる．

    引数:
        path (str): 動画のパス．
    戻り値:
        info (dict): "fps"，"width"，"height"，"duration"，"has_audio"．
    """
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError("動画として読み込めないファイルです．")
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("映像が含まれていないファイルです．")
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        numerator, _, denominator = video.get(key, "0/1").partition("/")
        if float(denominator or 1) > 0 and 1.0 <= float(numerator) / float(denominator or 1) <= 120.0:
            fps = float(numerator) / float(denominator or 1)
            break
    return {
        "fps": fps or 30.0, "width": int(video.get("width", 0)), "height": int(video.get("height", 0)),
        "duration": float(data.get("format", {}).get("duration", 0.0) or 0.0),
        "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


def normalize_video(source_path: str, output_path: str) -> dict:
    """
    動画を一定の fps，高さ・長さ・fps の上限内の H.264 の mp4 に変換する．ブラウザで撮影した WebM は
    fps が一定でない場合があるため，処理の前に必ずこの変換を行う．

    引数:
        source_path (str): 入力の動画のパス．
        output_path (str): 変換後の動画の保存先 (mp4)．
    戻り値:
        info (dict): 変換後の動画の情報 (`probe` の戻り値)．
    """
    info = probe(source_path)
    fps = min(round(info["fps"], 3), demo_settings.MAX_FPS)
    scale = f"scale=-2:'min({demo_settings.MAX_HEIGHT},ih)':flags=bicubic"
    command = ["ffmpeg", "-y", "-v", "error", "-i", source_path, "-t", str(demo_settings.MAX_DURATION_SEC),
               "-vf", f"{scale},fps={fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac", "-b:a", "128k"] if info["has_audio"] else ["-an"]
    run_command(command + ["-movflags", "+faststart", output_path])
    return probe(output_path)


def read_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    動画の全フレームを読み込む．

    引数:
        path (str): 動画のパス．
    戻り値:
        frames (list[np.ndarray]): 形状 (H, W, 3) のフレーム (BGR) のリスト．
        fps (float): フレームレート．
    """
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


class VideoAnonymizer:
    """
    モデルを一度だけ読み込み，動画を何本でも匿名化できるようにしたクラス．GPUは1つであるため，
    `process` は同時に1本ずつ呼び出すこと．
    """

    def __init__(self):
        """
        引数:
            なし．
        戻り値:
            なし．
        """
        from coordinate_reg.image_infer import Handler
        from identity_anonymizer import config

        threads = max(1, (os.cpu_count() or 1) // demo_settings.DETECTION_THREADS)
        self.models = ghost.load_models(with_dex=False, num_threads=threads)
        self.handler = Handler(os.path.join(config.GHOST_WEIGHTS_DIR, "2d106det"), 0, ctx_id=-1, det_size=640,
                               root=config.GHOST_WEIGHTS_DIR)
        self.vae = load_model()
        self.ranking = methods.load_ranking()
        # 実験時の再現性のための設定 (決定論的なアルゴリズム) は遅いため，デモでは外す
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def detect_all(self, frames: List[np.ndarray], report: ProgressCallback) -> List[List[np.ndarray]]:
        """
        全フレームで全ての顔を検出し，5点のランドマークを返す．顔検出器はCPUで動くため，スレッドで並列化する．

        引数:
            frames (list[np.ndarray]): 要素数 F のフレームのリスト．
            report (ProgressCallback): 進捗を報告する関数．
        戻り値:
            kps_per_frame (list[list[np.ndarray]]): 要素数 F．各要素は，そのフレームで検出した顔の
                ランドマーク (形状 (5, 2)) のリスト．
        """
        def detect(frame: np.ndarray) -> List[np.ndarray]:
            kps = self.models.app.get(frame, self.models.crop_size)
            return [] if kps is None else list(kps)

        results: List[List[np.ndarray]] = []
        with ThreadPoolExecutor(max_workers=demo_settings.DETECTION_THREADS) as pool:
            for i, kps in enumerate(pool.map(detect, frames)):
                results.append(kps)
                if i % 10 == 0:
                    report("detect", (i + 1) / len(frames))
        return results

    @torch.no_grad()
    def build_tracks(self, frames: List[np.ndarray], kps_per_frame: List[List[np.ndarray]],
                     report: ProgressCallback) -> Tuple[List[FaceTrack], np.ndarray]:
        """
        検出した顔を人物ごとに追跡し，ランドマークを平滑化して顔を切り出す．

        引数:
            frames (list[np.ndarray]): 要素数 F のフレームのリスト．
            kps_per_frame (list): `detect_all` の戻り値．
            report (ProgressCallback): 進捗を報告する関数．
        戻り値:
            tracks (list[FaceTrack]): 人物ごとの情報 (要素数 P)．
            embeddings (np.ndarray): 形状 (P, 512) の人物ごとの顔ベクトル．
        """
        from insightface.utils import face_align
        from utils.inference.video_processing import smooth_landmarks

        crop_size = self.models.crop_size
        embeddings_per_frame = []
        for i, (frame, kps_list) in enumerate(zip(frames, kps_per_frame)):
            crops = []
            for kps in kps_list:
                matrix, _ = face_align.estimate_norm(kps, crop_size, mode="None")
                crops.append(cv2.warpAffine(frame, matrix, (crop_size, crop_size), borderValue=0.0))
            # VAEの学習時と同じく，RGBの順序の顔画像から顔ベクトルを計算する
            embeddings_per_frame.append(ghost.embed(self.models, np.array(crops)).cpu().numpy() if crops
                                        else np.zeros((0, 512), dtype=np.float32))
            if i % 20 == 0:
                report("track", 0.8 * (i + 1) / len(frames))

        track_ids = tracking.assign_tracks(embeddings_per_frame, demo_settings.TRACK_SIMILARITY_THRESHOLD, kps_per_frame)
        num_tracks = 1 + max((int(ids.max()) for ids in track_ids if len(ids) > 0), default=-1)
        if num_tracks == 0:
            return [], np.zeros((0, 512), dtype=np.float32)

        raw_kps = [[[] for _ in frames] for _ in range(num_tracks)]
        for i, (kps_list, ids) in enumerate(zip(kps_per_frame, track_ids)):
            for kps, track in zip(kps_list, ids):
                raw_kps[track][i] = kps
        smoothed = smooth_landmarks(raw_kps, n=2)

        tracks = []
        for track in range(num_tracks):
            face_track = FaceTrack()
            for i, frame in enumerate(frames):
                kps = smoothed[track][i] if len(raw_kps[track][i]) > 0 else []
                if len(kps) == 0:
                    face_track.kps.append([])
                    face_track.crops.append([])
                    face_track.transforms.append([])
                    continue
                matrix, _ = face_align.estimate_norm(kps, crop_size, mode="None")
                face_track.kps.append(kps)
                face_track.crops.append(cv2.warpAffine(frame, matrix, (crop_size, crop_size), borderValue=0.0))
                face_track.transforms.append(matrix)
            tracks.append(face_track)
        report("track", 1.0)
        return tracks, tracking.track_embeddings(embeddings_per_frame, track_ids)

    @torch.no_grad()
    def generate(self, tracks: List[FaceTrack], sources: torch.Tensor, report: ProgressCallback) -> None:
        """
        全ての人物の全ての顔をGHOSTで生成し，`FaceTrack.generated` に格納する．

        引数:
            tracks (list[FaceTrack]): 人物ごとの情報．
            sources (torch.Tensor): 形状 (P, 512) の人物ごとのソースの顔ベクトル．
            report (ProgressCallback): 進捗を報告する関数．
        戻り値:
            なし．
        """
        items = [(t, i) for t, track in enumerate(tracks) for i, crop in enumerate(track.crops) if len(crop) > 0]
        for track in tracks:
            track.generated = [[] for _ in track.crops]
        batch_size = demo_settings.GENERATION_BATCH_SIZE
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            targets = np.stack([cv2.resize(tracks[t].crops[i], (256, 256)) for t, i in batch])
            batch_sources = sources[torch.tensor([t for t, _ in batch], device=sources.device)]
            for (t, i), image in zip(batch, ghost.generate(self.models, targets, batch_sources)):
                tracks[t].generated[i] = image
            report("generate", min(1.0, (start + batch_size) / max(len(items), 1)))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def paste(self, frames: List[np.ndarray], tracks: List[FaceTrack], output_path: str, fps: float,
              report: ProgressCallback) -> None:
        """
        生成した顔を元のフレームへ貼り戻し，動画として書き出す．GHOSTの `get_final_video` と同じ手順
        (ランドマークによるマスク，人物ごとのマスクのパラメータの保持，GPUでのアフィン変換と合成) を，
        進捗を報告できるようにしたものである．

        引数:
            frames (list[np.ndarray]): 要素数 F の元のフレーム．
            tracks (list[FaceTrack]): 人物ごとの情報 (生成済み)．
            output_path (str): 書き出す動画のパス (mp4v)．
            fps (float): フレームレート．
            report (ProgressCallback): 進捗を報告する関数．
        戻り値:
            なし．
        """
        import kornia
        from utils.inference.masks import face_mask_static

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        params: List[Optional[list]] = [None for _ in tracks]
        target_landmarks: List[Optional[np.ndarray]] = [None for _ in tracks]
        for i, frame in enumerate(frames):
            result = frame
            for j, track in enumerate(tracks):
                if len(track.crops[i]) == 0:
                    params[j] = None
                    continue
                swap = cv2.resize(track.generated[i], (224, 224))
                landmarks = self.handler.get_without_detection_without_transform(swap)
                if params[j] is None:
                    target_landmarks[j] = self.handler.get_without_detection_without_transform(track.crops[i])
                    mask, params[j] = face_mask_static(swap, landmarks, target_landmarks[j], params[j])
                else:
                    mask = face_mask_static(swap, landmarks, target_landmarks[j], params[j])
                swap_t = torch.from_numpy(swap).cuda().permute(2, 0, 1).unsqueeze(0).float()
                mask_t = torch.from_numpy(mask).cuda().unsqueeze(0).unsqueeze(0).float()
                full = torch.from_numpy(np.ascontiguousarray(result)).cuda().permute(2, 0, 1).unsqueeze(0).float()
                matrix = torch.from_numpy(track.transforms[i]).cuda().unsqueeze(0).float()
                inverse = kornia.invert_affine_transform(matrix)
                swap_t = kornia.warp_affine(swap_t, inverse, (height, width))
                mask_t = kornia.warp_affine(mask_t, inverse, (height, width))
                result = (mask_t * swap_t + (1 - mask_t) * full).type(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
            writer.write(result)
            if i % 10 == 0:
                report("paste", (i + 1) / len(frames))
        writer.release()

    def process(self, input_path: str, work_dir: str, method_id: str, params: Dict[str, float], seed: int,
                fix_norm: bool, report: ProgressCallback) -> dict:
        """
        動画1本を匿名化する．

        引数:
            input_path (str): 入力の動画のパス (ブラウザで撮影した WebM，任意の動画ファイル)．
            work_dir (str): 中間ファイルと出力を置くディレクトリ．
            method_id (str): 方式の識別子．
            params (dict): 方式のパラメータ．
            seed (int): 乱数シード．
            fix_norm (bool): ソースの顔ベクトルのノルムを実データの平均に補正するかどうか．
            report (ProgressCallback): 進捗を報告する関数 (処理段の名前, その処理段の中での割合)．
        戻り値:
            summary (dict): 出力のパス，動画の情報，人物の数，検出した顔の数，処理段ごとの所要時間など．
        """
        params = methods.validate(method_id, params)
        timings: Dict[str, float] = {}
        start = time.perf_counter()

        normalized_path = os.path.join(work_dir, "original.mp4")
        if not os.path.exists(normalized_path):
            normalize_video(input_path, normalized_path)
        report("normalize", 1.0)
        info = probe(normalized_path)
        frames, fps = read_frames(normalized_path)
        if len(frames) == 0:
            raise ValueError("フレームを読み込めませんでした．")
        timings["normalize"] = time.perf_counter() - start

        start = time.perf_counter()
        kps_per_frame = self.detect_all(frames, report)
        timings["detect"] = time.perf_counter() - start

        start = time.perf_counter()
        tracks, embeddings = self.build_tracks(frames, kps_per_frame, report)
        timings["track"] = time.perf_counter() - start

        anonymized_path = os.path.join(work_dir, "anonymized.mp4")
        silent_path = os.path.join(work_dir, "anonymized_silent.mp4")
        start = time.perf_counter()
        if tracks:
            sources = methods.make_sources(self.vae, torch.from_numpy(embeddings).cuda(), method_id, params,
                                           seed, fix_norm, self.ranking)
            self.generate(tracks, sources, report)
            source_norms = sources.norm(dim=1).cpu().numpy().tolist()
            source_cosines = torch.nn.functional.cosine_similarity(
                sources, torch.from_numpy(embeddings).cuda(), dim=1).cpu().numpy().tolist()
        else:
            source_norms, source_cosines = [], []
        timings["generate"] = time.perf_counter() - start

        start = time.perf_counter()
        self.paste(frames, tracks, silent_path, fps, report)
        timings["paste"] = time.perf_counter() - start

        start = time.perf_counter()
        command = ["ffmpeg", "-y", "-v", "error", "-i", silent_path]
        if info["has_audio"]:
            command += ["-i", normalized_path, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", anonymized_path]
        run_command(command)
        os.remove(silent_path)
        report("encode", 1.0)
        timings["encode"] = time.perf_counter() - start

        faces_per_track = [sum(len(crop) > 0 for crop in track.crops) for track in tracks]
        return {
            "method": method_id, "params": params, "seed": int(seed), "fix_norm": bool(fix_norm),
            "num_frames": len(frames), "fps": fps, "width": info["width"], "height": info["height"],
            "duration": info["duration"], "has_audio": info["has_audio"],
            "num_people": len(tracks), "faces_per_person": faces_per_track,
            "num_faces": int(sum(len(k) for k in kps_per_frame)),
            "frames_with_faces": int(sum(len(k) > 0 for k in kps_per_frame)),
            "cos_source_vs_original": source_cosines, "source_norms": source_norms,
            "timings_sec": timings, "total_sec": float(sum(timings.values())),
        }
