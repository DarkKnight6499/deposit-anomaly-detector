import unittest

import pandas as pd

from deposit_anomaly import calendar as cal


def frame():
    return cal.calendar_frame(pd.date_range("2022-01-01", "2023-12-31"))


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.f = frame()

    def test_weekend_and_holiday_not_business(self):
        self.assertFalse(self.f.loc["2022-07-04", cal.IS_BUSINESS])  # Independence Day, a Monday
        self.assertTrue(self.f.loc["2022-07-04", cal.IS_HOLIDAY])
        self.assertTrue(self.f.loc["2022-07-02", cal.IS_WEEKEND])
        self.assertTrue(self.f.loc["2022-07-05", cal.IS_BUSINESS])

    def test_month_end_is_last_business_day(self):
        self.assertTrue(self.f.loc["2022-04-29", cal.MONTH_END])  # 30 Apr 2022 is a Saturday
        self.assertFalse(self.f.loc["2022-04-30", cal.MONTH_END])
        self.assertEqual(int(self.f[cal.MONTH_END].sum()), 24)

    def test_first_business_day(self):
        self.assertTrue(self.f.loc["2022-05-02", cal.FIRST_BUSINESS])  # 1 May is a Sunday

    def test_payroll_rolls_back_to_prior_business_day(self):
        self.assertTrue(self.f.loc["2022-04-15", cal.PAYROLL])
        self.assertTrue(self.f.loc["2022-05-13", cal.PAYROLL])  # 15 May is a Sunday
        self.assertFalse(self.f.loc["2022-05-15", cal.PAYROLL])

    def test_tax_day_rolls_forward(self):
        self.assertTrue(self.f.loc["2023-04-17", cal.TAX_DAY])  # 15 Apr 2023 is a Saturday
        self.assertTrue(self.f.loc["2022-06-15", cal.TAX_DAY])
        self.assertEqual(int(self.f.loc["2022", cal.TAX_DAY].sum()), 4)

    def test_key_separates_special_days(self):
        self.assertNotEqual(self.f.loc["2022-04-15", cal.DAY_KEY], self.f.loc["2022-04-14", cal.DAY_KEY])
        self.assertEqual(self.f.loc["2022-07-02", cal.DAY_KEY], "weekend")

    def test_flags_use_date_only(self):
        sub = cal.calendar_frame(pd.date_range("2022-03-01", "2022-04-30"))
        pd.testing.assert_series_equal(sub.loc["2022-03-01":"2022-04-20", cal.PAYROLL],
                                       self.f.loc["2022-03-01":"2022-04-20", cal.PAYROLL])


if __name__ == "__main__":
    unittest.main()
