"""Speaker/text disjoint splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


class SpeakerDisjointSplitter:
    def __init__(self, speakers: Sequence[str], seed: int = 1337):
        self.speakers = np.array(speakers)
        self.seed = seed

    def train_test_split(self, test_size: float = 0.25) -> SplitIndices:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        idx = np.arange(len(self.speakers))
        (train_idx, test_idx), = gss.split(idx, groups=self.speakers)
        return SplitIndices(train=train_idx, test=test_idx)

    def kfold(self, n_splits: int = 5) -> Iterable[SplitIndices]:
        gkf = GroupKFold(n_splits=n_splits)
        idx = np.arange(len(self.speakers))
        for train_idx, test_idx in gkf.split(idx, groups=self.speakers):
            yield SplitIndices(train=train_idx, test=test_idx)


def text_disjoint_split(text_ids: Sequence[str], test_size: float = 0.2, seed: int = 1337) -> SplitIndices:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = np.arange(len(text_ids))
    (train_idx, test_idx), = gss.split(indices, groups=text_ids)
    return SplitIndices(train=train_idx, test=test_idx)
