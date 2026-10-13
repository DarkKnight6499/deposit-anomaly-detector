"""Column and label name constants shared by the generator, features and models."""
from __future__ import annotations

SEGMENTS: tuple[str, ...] = (
    "retail_insured",
    "retail_uninsured",
    "small_business",
    "corp_operational",
    "corp_nonoperational",
    "brokered",
)


def balance_col(segment: str) -> str:
    return f"balance_{segment}"


def flow_col(segment: str) -> str:
    return f"flow_{segment}"


BALANCE_COLUMNS = tuple(balance_col(s) for s in SEGMENTS)
FLOW_COLUMNS = tuple(flow_col(s) for s in SEGMENTS)

# Observed market and operational indicator columns.
BANK_CDS = "bank_cds_bps"
MARKET_CDS = "market_cds_bps"
BANK_EQUITY = "bank_equity_px"
MARKET_EQUITY = "market_equity_px"
RATE_2Y = "rate_2y_pct"
RATE_SPREAD = "competitor_rate_spread_bps"
TOP10_SHARE = "top10_outflow_share"
RECON_GAP = "recon_gap_pct"
PAYMENT_QUEUE = "payment_queue_pct"

OBSERVED_EXTRA_COLUMNS = (
    BANK_CDS, MARKET_CDS, BANK_EQUITY, MARKET_EQUITY, RATE_2Y, RATE_SPREAD,
    TOP10_SHARE, RECON_GAP, PAYMENT_QUEUE,
)

# Label columns (kept in a separate frame from observed data).
EVENT_TYPE = "event_type"
EVENT_ID = "event_id"
Y_LIQUIDITY = "y_liquidity"
Y_DQ = "y_data_quality"
NO_EVENT = "none"

RUN = "idiosyncratic_run"
FLIGHT = "flight_to_quality"
BLEED = "slow_bleed"
GLITCH = "data_glitch"
OUTAGE = "payment_outage"
LIQUIDITY_TYPES = (RUN, FLIGHT, BLEED)
DQ_TYPES = (GLITCH, OUTAGE)
ALL_EVENT_TYPES = LIQUIDITY_TYPES + DQ_TYPES

# Events table columns.
EV_START = "start"
EV_END = "end"
EV_CRITICAL = "critical_day"
EV_OUTFLOW_START = "outflow_start"
EV_TOTAL_OUTFLOW = "extra_outflow_pct_of_total_balance"
