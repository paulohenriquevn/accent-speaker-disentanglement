"""Feature extraction modules."""

from .acoustic import AcousticFeatureExtractor
from .ecapa import ECAPAExtractor
from .ssl import SSLFeatureExtractor
from .backbone import BackboneFeatureExtractor

__all__ = [
    "AcousticFeatureExtractor",
    "ECAPAExtractor",
    "SSLFeatureExtractor",
    "BackboneFeatureExtractor",
]
