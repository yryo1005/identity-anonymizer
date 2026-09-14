import os
from dataclasses import dataclass

import torch

from identity_anonymizer import config

# third_party/ghost を sys.path に通してから，GHOST側のモジュールをimportする
from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions


@dataclass
class GhostModels:
    """
    GHOSTによる顔交換に必要な各モデルをまとめて保持するデータクラス．

    属性:
        app (Face_detect_crop): 顔検出・切り出しモデル．
        generator (AEI_Net): GHOSTの顔交換生成器．
        arcface (iresnet100): ArcFaceによる顔ベクトル抽出モデル．
        handler (Handler): 顔ランドマーク検出モデル．
        super_resolution (Pix2PixModel または None): 超解像モデル(use_sr=Falseの場合はNone)．
        crop_size (int): GHOSTが前提とする顔画像の一辺のサイズ(224固定)．
    """

    app: Face_detect_crop
    generator: AEI_Net
    arcface: torch.nn.Module
    handler: Handler
    super_resolution: object
    crop_size: int = 224


def load_ghost_models(
    weights_dir: str = None,
    use_sr: bool = False,
    det_thresh: float = 0.6,
    det_size: tuple = (640, 640),
) -> GhostModels:
    """
    GHOSTによる顔交換に必要な各モデル(顔検出，生成器，ArcFace，顔ランドマーク検出，超解像)を
    読み込み，GPU上に配置する．

    引数:
        weights_dir (str または None): GHOST由来の重みを格納したディレクトリ．
            None の場合は `identity_anonymizer.config.GHOST_WEIGHTS_DIR` を使用する．
        use_sr (bool): 超解像モデルを読み込むかどうか．Falseの場合は起動を高速化するために
            読み込みをスキップし，`GhostModels.super_resolution` は None になる．
        det_thresh (float): 顔検出の信頼度閾値．
        det_size (tuple[int, int]): 顔検出時の入力解像度．
    戻り値:
        models (GhostModels): 読み込み済みモデル一式．
    """
    if weights_dir is None:
        weights_dir = config.GHOST_WEIGHTS_DIR

    os.environ.setdefault("MXNET_USE_FUSION", "0")

    # Face_detect_crop は "<root>/<name>/*.onnx" を探すため，antelopeモデルは
    # <weights_dir>/antelope/ に配置しておく必要がある
    app = Face_detect_crop(name="antelope", root=weights_dir)
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    generator = AEI_Net(backbone="unet", num_blocks=2, c_id=512)
    generator.eval()
    generator.load_state_dict(torch.load(os.path.join(weights_dir, "G_unet_2blocks.pth"), map_location="cpu"))
    generator = generator.cuda().half()

    arcface = iresnet100(fp16=False)
    arcface.load_state_dict(torch.load(os.path.join(weights_dir, "backbone.pth"), map_location="cpu"))
    arcface = arcface.cuda()
    arcface.eval()

    # Handler は内部で独自に Face_detect_crop(name='antelope', root=root) を生成するため，
    # antelopeモデルの配置場所(weights_dir)を root として明示的に渡す必要がある
    handler = Handler(os.path.join(weights_dir, "2d106det"), 0, ctx_id=0, det_size=640, root=weights_dir)

    super_resolution = None
    if use_sr:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        super_resolution = Pix2PixModel(opt)
        super_resolution.netG.train()

    return GhostModels(
        app=app,
        generator=generator,
        arcface=arcface,
        handler=handler,
        super_resolution=super_resolution,
    )
