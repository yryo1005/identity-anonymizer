"""
動画の匿名化デモのWebサーバー．

ブラウザで撮影した動画，またはローカルの動画ファイルを受け取り，選んだ方式で動画中の全員の顔を
匿名化して，ブラウザで再生できる動画として返す．GPUは1つであるため，ジョブは1本ずつ順番に処理する．

    source programs/demo/env.sh
    $PYTHON programs/demo/app.py --host 127.0.0.1 --port 8000

ブラウザで http://127.0.0.1:8000 を開く．カメラでの撮影には安全なコンテキスト (localhost または HTTPS) が必要である．
"""

import argparse
import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from typing import Dict, Optional

import demo_settings
import methods

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STAGE_LABELS = {
    "queued": "順番待ち", "normalize": "動画を変換中", "detect": "顔を検出中", "track": "人物を追跡中",
    "generate": "顔を生成中", "paste": "顔を貼り戻し中", "encode": "動画を書き出し中",
    "done": "完了", "error": "エラー",
}


class JobManager:
    """
    匿名化のジョブを受け付け，1本ずつ順番に処理し，状態を保持するクラス．
    """

    def __init__(self):
        """
        引数:
            なし．
        戻り値:
            なし．
        """
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.anonymizer = None
        self.model_error: Optional[str] = None
        os.makedirs(demo_settings.JOBS_DIR, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def _load_models(self) -> None:
        """
        モデルを読み込む (サーバーの起動直後に，ワーカーのスレッドで1回だけ実行する)．

        引数:
            なし．
        戻り値:
            なし．
        """
        try:
            from engine import VideoAnonymizer
            self.anonymizer = VideoAnonymizer()
        except Exception as error:  # 読み込みの失敗はブラウザに表示する
            self.model_error = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    def _worker(self) -> None:
        """
        ジョブを1本ずつ取り出して処理するワーカー．

        引数:
            なし．
        戻り値:
            なし．
        """
        self._load_models()
        while True:
            job_id = self.queue.get()
            self._run(job_id)

    def _update(self, job_id: str, **fields) -> None:
        """
        ジョブの状態を更新する．

        引数:
            job_id (str): ジョブの識別子．
            **fields: 更新する項目．
        戻り値:
            なし．
        """
        with self.lock:
            self.jobs[job_id].update(fields)

    def _run(self, job_id: str) -> None:
        """
        1本のジョブを処理する．

        引数:
            job_id (str): ジョブの識別子．
        戻り値:
            なし．
        """
        from engine import STAGE_WEIGHTS

        job = self.jobs[job_id]
        if self.anonymizer is None:
            self._update(job_id, stage="error", error=f"モデルを読み込めませんでした: {self.model_error}")
            return
        stages = list(STAGE_WEIGHTS)

        def report(stage: str, fraction: float) -> None:
            """処理段とその中での割合から，全体の進捗を計算して記録する．"""
            done = sum(STAGE_WEIGHTS[s] for s in stages[:stages.index(stage)])
            self._update(job_id, stage=stage, progress=min(0.99, done + STAGE_WEIGHTS[stage] * fraction))

        try:
            started = time.time()
            summary = self.anonymizer.process(job["input_path"], job["dir"], job["method"], job["params"],
                                              job["seed"], job["fix_norm"], report)
            with open(os.path.join(job["dir"], "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            self._update(job_id, stage="done", progress=1.0, summary=summary, finished_at=time.time(),
                         elapsed_sec=time.time() - started)
        except Exception as error:
            traceback.print_exc()
            self._update(job_id, stage="error", error=f"{type(error).__name__}: {error}")

    def submit(self, input_path: str, job_dir: str, method_id: str, params: dict, seed: int, fix_norm: bool,
               input_name: str) -> str:
        """
        ジョブを登録して処理の順番待ちに入れる．

        引数:
            input_path (str): 入力の動画のパス．
            job_dir (str): ジョブのディレクトリ．
            method_id (str): 方式の識別子．
            params (dict): 方式のパラメータ．
            seed (int): 乱数シード．
            fix_norm (bool): ノルムを補正するかどうか．
            input_name (str): 入力の表示名．
        戻り値:
            job_id (str): ジョブの識別子．
        """
        job_id = os.path.basename(job_dir)
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id, "dir": job_dir, "input_path": input_path, "input_name": input_name,
                "method": method_id, "params": methods.validate(method_id, params), "seed": seed,
                "fix_norm": fix_norm, "stage": "queued", "progress": 0.0, "created_at": time.time(),
            }
        self._cleanup()
        self.queue.put(job_id)
        return job_id

    def _cleanup(self) -> None:
        """
        保持するジョブの数を超えた場合に，古い完了済みのジョブのファイルを削除する．

        引数:
            なし．
        戻り値:
            なし．
        """
        with self.lock:
            finished = sorted((j for j in self.jobs.values() if j["stage"] in ("done", "error")), key=lambda j: j["created_at"])
            excess = len(self.jobs) - demo_settings.MAX_KEPT_JOBS
            for job in finished[:max(0, excess)]:
                shutil.rmtree(job["dir"], ignore_errors=True)
                del self.jobs[job["id"]]

    def public(self, job_id: str) -> dict:
        """
        ブラウザへ返すためのジョブの状態を作る (ファイルのパスなどの内部情報は含めない)．

        引数:
            job_id (str): ジョブの識別子．
        戻り値:
            state (dict): ジョブの状態．
        """
        with self.lock:
            job = dict(self.jobs[job_id])
        position = list(self.queue.queue).index(job_id) + 1 if job_id in list(self.queue.queue) else 0
        return {
            "id": job["id"], "stage": job["stage"], "stage_label": STAGE_LABELS.get(job["stage"], job["stage"]),
            "progress": job["progress"], "queue_position": position, "method": job["method"],
            "params": job["params"], "seed": job["seed"], "fix_norm": job["fix_norm"], "input_name": job["input_name"],
            "summary": job.get("summary"), "error": job.get("error"), "elapsed_sec": job.get("elapsed_sec"),
        }


app = FastAPI(title="Identity Anonymizer Demo")
manager: Optional[JobManager] = None


def get_manager() -> JobManager:
    """
    ジョブの管理クラスを返す (初回の呼び出しで作成する)．

    引数:
        なし．
    戻り値:
        manager (JobManager): ジョブの管理クラス．
    """
    global manager
    if manager is None:
        manager = JobManager()
    return manager


@app.on_event("startup")
def startup() -> None:
    """
    サーバーの起動時に，ジョブの管理クラスを作成してモデルの読み込みを始める．

    引数:
        なし．
    戻り値:
        なし．
    """
    get_manager()


@app.get("/api/status")
def status() -> dict:
    """
    モデルの読み込みの状態と，入力の制限を返す．

    引数:
        なし．
    戻り値:
        state (dict): "ready" (bool)，"error" (str または None)，"limits" (dict)．
    """
    current = get_manager()
    return {
        "ready": current.anonymizer is not None, "error": current.model_error,
        "limits": {"max_duration_sec": demo_settings.MAX_DURATION_SEC, "max_height": demo_settings.MAX_HEIGHT,
                   "max_upload_mb": demo_settings.MAX_UPLOAD_BYTES // (1024 * 1024)},
    }


@app.get("/api/methods")
def list_methods() -> list:
    """
    選べる方式の一覧を返す．

    引数:
        なし．
    戻り値:
        methods (list[dict]): 方式の識別子，表示名，説明，パラメータ，プリセット．
    """
    return methods.METHODS


async def save_upload(upload: UploadFile, path: str) -> None:
    """
    アップロードされたファイルを，サイズの上限を確かめながら保存する．

    引数:
        upload (UploadFile): アップロードされたファイル．
        path (str): 保存先のパス．
    戻り値:
        なし．
    """
    size = 0
    with open(path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > demo_settings.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="ファイルが大きすぎます．")
            f.write(chunk)


@app.post("/api/jobs")
async def create_job(method: str = Form(...), params: str = Form("{}"), seed: int = Form(0),
                     fix_norm: bool = Form(False), source_job: str = Form(""),
                     file: Optional[UploadFile] = File(None)) -> dict:
    """
    匿名化のジョブを登録する．動画は新たにアップロードするか，以前のジョブの入力を再利用する．

    引数:
        method (str): 方式の識別子．
        params (str): 方式のパラメータ (JSON)．
        seed (int): 乱数シード．
        fix_norm (bool): ノルムを補正するかどうか．
        source_job (str): 入力を再利用する以前のジョブの識別子 (空の場合は file を用いる)．
        file (UploadFile または None): アップロードされた動画．
    戻り値:
        job (dict): 登録したジョブの状態．
    """
    current = get_manager()
    try:
        parsed = methods.validate(method, json.loads(params or "{}"))
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(demo_settings.JOBS_DIR, job_id)
    os.makedirs(job_dir)
    if source_job:
        with current.lock:
            previous = current.jobs.get(source_job)
        if previous is None or not os.path.exists(os.path.join(previous["dir"], "original.mp4")):
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=404, detail="再利用する動画が見つかりません．")
        shutil.copy(os.path.join(previous["dir"], "original.mp4"), os.path.join(job_dir, "original.mp4"))
        input_path, input_name = os.path.join(job_dir, "original.mp4"), previous["input_name"]
    elif file is not None:
        extension = os.path.splitext(file.filename or "")[1].lower()[:8] or ".bin"
        input_path, input_name = os.path.join(job_dir, f"input{extension}"), file.filename or "upload"
        try:
            await save_upload(file, input_path)
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
    else:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="動画が指定されていません．")

    current.submit(input_path, job_dir, method, parsed, int(seed), bool(fix_norm), input_name)
    return current.public(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """
    ジョブの状態を返す．

    引数:
        job_id (str): ジョブの識別子．
    戻り値:
        job (dict): ジョブの状態．
    """
    current = get_manager()
    if job_id not in current.jobs:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません．")
    return current.public(job_id)


@app.get("/api/jobs/{job_id}/video/{kind}")
def get_video(job_id: str, kind: str, download: bool = False):
    """
    ジョブの動画 (変換後の元の動画，または匿名化後の動画) を返す．

    引数:
        job_id (str): ジョブの識別子．
        kind (str): "original" または "anonymized"．
        download (bool): True の場合，ダウンロード用のファイル名を付ける．
    戻り値:
        response (FileResponse): 動画のファイル．
    """
    current = get_manager()
    if job_id not in current.jobs or kind not in ("original", "anonymized"):
        raise HTTPException(status_code=404, detail="動画が見つかりません．")
    path = os.path.join(current.jobs[job_id]["dir"], f"{kind}.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="動画はまだ作成されていません．")
    filename = f"{kind}_{job_id}.mp4" if download else None
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/")
def index() -> FileResponse:
    """
    デモの画面を返す．

    引数:
        なし．
    戻り値:
        response (FileResponse): `static/index.html`．
    """
    return FileResponse(os.path.join(demo_settings.STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=demo_settings.STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="動画の匿名化デモのWebサーバーを起動する")
    parser.add_argument("--host", default="127.0.0.1", help="待ち受けるアドレス (外部に公開する場合のみ 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")
