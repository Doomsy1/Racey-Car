"""Top-level package exports for the vision pipeline.

This module exposes the main public classes and helpers so callers can do:

	from vision_pipeline import run

The package's CLI entrypoint remains available as ``vision_pipeline.run.main``.
"""

from .run import main

__all__ = [
	"main",
]
