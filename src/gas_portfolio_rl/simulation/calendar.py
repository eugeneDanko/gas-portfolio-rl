from __future__ import annotations

import calendar
from datetime import date, timedelta
import math


def model_date(trading_day: int, start_year: int, trading_days: int) -> date:
    """Map a model trading index to a real calendar label with one model year per cycle."""
    if trading_day < 0:
        raise ValueError("trading_day must be non-negative")
    year_offset, day_in_model_year = divmod(trading_day, trading_days)
    year = start_year + year_offset
    calendar_days = 366 if calendar.isleap(year) else 365
    day_of_year = min(
        int(round(day_in_model_year * calendar_days / trading_days)),
        calendar_days - 1,
    )
    return date(year, 1, 1) + timedelta(days=day_of_year)


def model_trading_day(value: date, start_year: int, trading_days: int) -> int:
    if value.year < start_year:
        raise ValueError("value must not precede start_year")
    calendar_days = 366 if calendar.isleap(value.year) else 365
    day_of_year = value.timetuple().tm_yday - 1
    day_in_model_year = min(
        int(round(day_of_year * trading_days / calendar_days)),
        trading_days - 1,
    )
    return (value.year - start_year) * trading_days + day_in_model_year


def next_contract_date(current: date, expiry_day: int) -> date:
    candidate = date(current.year, current.month, expiry_day)
    return candidate if current < candidate else add_months(candidate, 1)


def crossed_monthly_date(previous: date, current: date, day: int) -> bool:
    """Return whether ``(month, day)`` was crossed in ``(previous, current]``."""
    if current <= previous:
        return False
    candidate = date(previous.year, previous.month, day)
    if candidate <= previous:
        candidate = add_months(candidate, 1)
    return candidate <= current


def add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + (value.month - 1) + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def cyclical_date_features(value: date) -> tuple[float, float]:
    days = 366 if calendar.isleap(value.year) else 365
    angle = 2.0 * math.pi * (value.timetuple().tm_yday - 1) / days
    return math.sin(angle), math.cos(angle)
