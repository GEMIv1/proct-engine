from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseDetector(ABC):
    """Abstract base class for all computer vision/audio detectors in the proctoring system.

    Provides standard context manager methods (__enter__, __exit__) and defines the
    common `process` signature for frame-by-frame or window-based analysis.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Clean up any detector resources (e.g. MediaPipe pipelines, GPU models)."""
        pass

    @abstractmethod
    def process(self, frame: np.ndarray, **kwargs: Any) -> Any:
        """Process a single frame or input window and return the detection result.

        Parameters
        ----------
        frame : np.ndarray
            The image frame or input matrix to process.
        **kwargs : Any
            Additional detector-specific parameters.

        Returns
        -------
        Any
            The result of detection (e.g. bool, GazeState, etc.).
        """
        pass
