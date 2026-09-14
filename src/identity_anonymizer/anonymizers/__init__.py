from identity_anonymizer.anonymizers.base import Anonymizer
from identity_anonymizer.anonymizers.registry import get_anonymizer, list_anonymizers, register_anonymizer
from identity_anonymizer.anonymizers.vae import VAEAnonymizer

__all__ = [
    "Anonymizer",
    "VAEAnonymizer",
    "get_anonymizer",
    "register_anonymizer",
    "list_anonymizers",
]
