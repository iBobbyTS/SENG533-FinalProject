from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    benchmark: str
    profile: str
    base_url: str
    model: str
    seed: int
    run_id: str
    output_root: Path
    data_root: Path
    cache_root: Path
    external_root: Path
