import pytest
import torch

from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.anonymizers.registry import get_anonymizer, list_anonymizers, register_anonymizer
from identity_anonymizer.anonymizers.vae import VAEAnonymizer


def test_vae_is_registered_by_default():
    assert "vae" in list_anonymizers()


def test_get_anonymizer_returns_vae_instance():
    model = get_anonymizer("vae", latent_dim=16, seed=0)
    assert isinstance(model, VAEAnonymizer)
    assert isinstance(model, Anonymizer)


def test_get_anonymizer_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        get_anonymizer("no-such-model")


def test_anonymize_preserves_output_shape():
    model = get_anonymizer("vae", latent_dim=16, seed=0)
    embedding = torch.randn(4, model.input_dim)
    anonymized = model.anonymize(embedding, noise_level=1.0)
    assert anonymized.shape == (4, model.output_dim)


def test_vae_validate_input_rejects_wrong_last_dim():
    model = get_anonymizer("vae", latent_dim=16, seed=0)
    wrong_shaped = torch.randn(4, model.input_dim + 1)
    with pytest.raises(ValueError):
        model.anonymize(wrong_shaped)


def test_vae_validate_input_rejects_wrong_type():
    model = get_anonymizer("vae", latent_dim=16, seed=0)
    with pytest.raises(TypeError):
        model.anonymize({"gender": "male", "age": 30})


class _AttributeAnonymizer(Anonymizer):
    """
    references/ の予稿で紹介されている従来手法(性別・年齢・人種等の属性を入力とする手法)を
    模した，入力形式がテンソルではないAnonymizerのテスト用実装．
    出力の次元数(output_dim)はAnonymizer側の規約に従い512に固定される一方，
    入力はdict(属性名 -> 値)を受け取る点がVAEAnonymizerと異なる．
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, self.output_dim)

    def validate_input(self, x):
        if not isinstance(x, dict) or "age" not in x:
            raise TypeError(f"_AttributeAnonymizerの入力はキー'age'を含むdictである必要があります: {x}")

    def _anonymize(self, x, **kwargs):
        age_tensor = torch.tensor([[float(x["age"])]])
        return self.linear(age_tensor)


def test_register_anonymizer_allows_non_tensor_input():
    register_anonymizer("attribute_dummy", _AttributeAnonymizer)
    assert "attribute_dummy" in list_anonymizers()

    model = get_anonymizer("attribute_dummy")
    output = model.anonymize({"age": 30, "gender": "male"})
    assert output.shape == (1, model.output_dim)

    with pytest.raises(TypeError):
        model.anonymize({"gender": "male"})  # "age" キーが無い


def test_register_anonymizer_rejects_non_anonymizer_class():
    class NotAnAnonymizer:
        pass

    with pytest.raises(TypeError):
        register_anonymizer("invalid", NotAnAnonymizer)
