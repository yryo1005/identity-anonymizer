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


def test_anonymize_preserves_shape():
    model = get_anonymizer("vae", latent_dim=16, seed=0)
    embedding = torch.randn(4, model.embedding_dim)
    anonymized = model.anonymize(embedding, noise_level=1.0)
    assert anonymized.shape == embedding.shape


def test_register_anonymizer_allows_swapping_models():
    class DummyAnonymizer(Anonymizer):
        def __init__(self):
            super().__init__()

        def anonymize(self, embedding, **kwargs):
            return embedding  # 恒等写像(テスト用のダミー実装)

    register_anonymizer("dummy", DummyAnonymizer)
    assert "dummy" in list_anonymizers()

    model = get_anonymizer("dummy")
    embedding = torch.randn(2, model.embedding_dim)
    assert torch.equal(model.anonymize(embedding), embedding)


def test_register_anonymizer_rejects_non_anonymizer_class():
    class NotAnAnonymizer:
        pass

    with pytest.raises(TypeError):
        register_anonymizer("invalid", NotAnAnonymizer)
