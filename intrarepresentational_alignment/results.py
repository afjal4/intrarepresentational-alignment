from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import PermutationTestResult


@dataclass
class DomainAlignmentResult:
    n_pairs: int
    cka: PermutationTestResult
    ged: PermutationTestResult
    K_source: np.ndarray | None = None
    K_target: np.ndarray | None = None
