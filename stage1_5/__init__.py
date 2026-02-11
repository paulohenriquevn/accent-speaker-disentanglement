"""Stage 1.5 latent separability audit toolkit."""

from importlib.metadata import version, PackageNotFoundError


try:  # pragma: no cover - defensive guard
    __version__ = version("stage1_5")
except PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.0.0"

__all__ = ["__version__"]
