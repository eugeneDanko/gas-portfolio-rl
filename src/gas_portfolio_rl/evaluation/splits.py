from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedManifest:
    """Non-overlapping random streams frozen before any PPO training."""

    calibration: tuple[int, int] = (0, 999)
    training: tuple[int, int] = (1_000, 9_999)
    validation: tuple[int, int] = (10_000, 19_999)
    test: tuple[int, int] = (20_000, 29_999)
    stress: tuple[int, int] = (30_000, 39_999)

    def validate(self) -> None:
        ranges = [self.calibration, self.training, self.validation, self.test, self.stress]
        for start, end in ranges:
            if start < 0 or end < start:
                raise ValueError("Invalid seed range")
        for index, left in enumerate(ranges):
            for right in ranges[index + 1:]:
                if max(left[0], right[0]) <= min(left[1], right[1]):
                    raise ValueError("Seed ranges must not overlap")

    def as_dict(self) -> dict[str, dict[str, int]]:
        self.validate()
        return {
            name: {"start": bounds[0], "end": bounds[1]}
            for name, bounds in (
                ("calibration", self.calibration),
                ("training", self.training),
                ("validation", self.validation),
                ("test", self.test),
                ("stress", self.stress),
            )
        }
