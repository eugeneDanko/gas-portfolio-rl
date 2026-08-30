from __future__ import annotations

import bisect
import calendar
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SeasonalityProfile:
    """Periodic piecewise-linear log-seasonality indexed by delivery date."""

    anchors: tuple[tuple[str, float], ...]
    amplitude_scale: float = 1.0
    phase_shift_days: int = 0

    def __post_init__(self) -> None:
        points = sorted((_anchor_position(key), float(value)) for key, value in self.anchors)
        object.__setattr__(self, "_positions", tuple(position for position, _ in points))
        object.__setattr__(self, "_values", tuple(value for _, value in points))

    def value(self, calendar_date: date) -> float:
        days = 366 if calendar.isleap(calendar_date.year) else 365
        position = (
            (calendar_date.timetuple().tm_yday - 1 - self.phase_shift_days) / days
        ) % 1.0
        positions = self._positions
        values = self._values
        extended_positions = (positions[-1] - 1.0,) + positions + (positions[0] + 1.0,)
        extended_values = (values[-1],) + values + (values[0],)
        right = bisect.bisect_right(extended_positions, position)
        left = right - 1
        x0, x1 = extended_positions[left], extended_positions[right]
        y0, y1 = extended_values[left], extended_values[right]
        weight = 0.0 if x1 == x0 else (position - x0) / (x1 - x0)
        return self.amplitude_scale * (y0 + weight * (y1 - y0))


def _anchor_position(month_day: str) -> float:
    month, day = (int(part) for part in month_day.split("-"))
    anchor = date(2001, month, day)
    return (anchor.timetuple().tm_yday - 1) / 365.0
