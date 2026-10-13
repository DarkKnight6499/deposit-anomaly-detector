"""Public calendar flags (known in advance, so using them at day t is not look-ahead)."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

IS_WEEKEND = "is_weekend"
IS_HOLIDAY = "is_holiday"
IS_BUSINESS = "is_business"
PAYROLL = "payroll_day"
MONTH_END = "month_end"
FIRST_BUSINESS = "first_business_day"
TAX_DAY = "tax_day"
SB_PAYDAY = "sb_payday"
HOLIDAY_SEASON = "holiday_season"
AFTER_PAYROLL = "after_payroll"
DAY_KEY = "calendar_key"  # weekday / weekend / holiday only
SPECIAL_FLAGS = (PAYROLL, MONTH_END, FIRST_BUSINESS, TAX_DAY, SB_PAYDAY, AFTER_PAYROLL)

TAX_MONTHS = (1, 4, 6, 9)


def _roll(day: pd.Timestamp, business: set, step: int, limit: int = 5) -> pd.Timestamp | None:
    for _ in range(limit):
        if day in business:
            return day
        day = day + pd.Timedelta(days=step)
    return None


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean calendar flags plus a categorical key used for seasonal baselines."""
    idx = pd.DatetimeIndex(dates)
    hol = USFederalHolidayCalendar().holidays(idx.min(), idx.max())
    out = pd.DataFrame(index=idx)
    out[IS_WEEKEND] = idx.dayofweek >= 5
    out[IS_HOLIDAY] = idx.isin(hol) & ~out[IS_WEEKEND]
    out[IS_BUSINESS] = ~(out[IS_WEEKEND] | out[IS_HOLIDAY])
    business = set(idx[out[IS_BUSINESS].to_numpy()])

    payroll, taxd = set(), set()
    for ym in pd.period_range(idx.min(), idx.max(), freq="M"):
        for dom in (1, 15):
            d = _roll(pd.Timestamp(ym.year, ym.month, dom), business, -1)
            if d is not None:
                payroll.add(d)
        if ym.month in TAX_MONTHS:
            d = _roll(pd.Timestamp(ym.year, ym.month, 15), business, 1)
            if d is not None:
                taxd.add(d)
    out[PAYROLL] = idx.isin(list(payroll))
    out[TAX_DAY] = idx.isin(list(taxd))

    bdays = out.index[out[IS_BUSINESS]]
    month_key = bdays.to_period("M")
    last = pd.Series(bdays, index=bdays).groupby(month_key).max()
    first = pd.Series(bdays, index=bdays).groupby(month_key).min()
    out[MONTH_END] = idx.isin(list(last.values))
    out[FIRST_BUSINESS] = idx.isin(list(first.values))
    fridays = out[IS_BUSINESS] & (idx.dayofweek == 4)
    out[SB_PAYDAY] = fridays & ((idx.map(lambda d: d.toordinal()) // 7) % 2 == 0)
    out[AFTER_PAYROLL] = out[PAYROLL].shift(1, fill_value=False) & out[IS_BUSINESS]
    out[HOLIDAY_SEASON] = (idx.month == 12) & (idx.day >= 15) & (idx.day <= 24)

    plain = pd.Series([f"dow{d}" for d in idx.dayofweek], index=idx).where(out[IS_BUSINESS], "weekend")
    out[DAY_KEY] = plain.where(~out[IS_HOLIDAY], "holiday")
    return out
