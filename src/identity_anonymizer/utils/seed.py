import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """
    実験の再現性を確保するため，Python/NumPy/PyTorch/CUDAのseedを固定する．

    引数:
        seed (int): 固定するseed値．デフォルトは0．
    戻り値:
        なし．
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
