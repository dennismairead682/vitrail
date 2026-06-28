from __future__ import annotations

from torch.utils.data import Dataset

from vitrail.atelier.schema import GlassSpec
from vitrail.cutline.synthetic import synth_paired
from vitrail.cutline.types import PairedSample


class GlazedCorpus(Dataset[PairedSample]):
    def __init__(
        self,
        spec: GlassSpec,
        length: int,
        size: tuple[int, int, int],
        teeth: int = 6,
        paired_fraction: float = 0.7,
        implant_fraction: float = 0.3,
        base_seed: int = 0,
    ) -> None:
        self.spec = spec
        self.length = length
        self.size = size
        self.teeth = teeth
        self.paired_fraction = paired_fraction
        self.implant_fraction = implant_fraction
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> PairedSample:
        seed = self.base_seed * 100003 + index
        paired = (index % 100) < int(self.paired_fraction * 100)
        implant = (index % 10) < int(self.implant_fraction * 10)
        return synth_paired(
            self.spec,
            self.size,
            self.teeth,
            seed,
            paired=paired,
            with_implant=implant,
        )
