from dataclasses import dataclass, field
from typing import Any
from .enums import ViolationType

@dataclass
class ViolationEntry:
    type: ViolationType
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class RecordingResult:
    filename: str
    frame_count: int = 0
    duration: float = 0.0
    fps: float = 0.0
