import numpy as np
import torch

from identity_anonymizer.evaluation.age_gender import _preprocess
from identity_anonymizer.evaluation.metrics import AnonymizationEvalResult


def test_preprocess_output_shape_and_dtype():
    image = np.zeros((100, 80, 3), dtype=np.uint8)
    tensor = _preprocess(image, torch.device("cpu"))

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_anonymization_eval_result_fields():
    result = AnonymizationEvalResult(
        image_path="dummy.jpg",
        original_age=30.0,
        anonymized_age=32.0,
        original_gender="Male",
        anonymized_gender="Male",
        cosine_similarity=0.8,
    )

    assert result.image_path == "dummy.jpg"
    assert result.original_gender == result.anonymized_gender == "Male"
    assert 0.0 <= result.cosine_similarity <= 1.0
