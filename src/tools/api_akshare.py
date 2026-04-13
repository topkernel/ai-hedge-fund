"""AKShare-based data provider for Chinese A-share stocks.

Implements the same API as the Financial Datasets provider, returning
identical Pydantic model instances so agents require no changes.
"""

import re
import os
import sys
import time
import logging
import math
import threading
from datetime import datetime

# Disable tqdm progress bars from AKShare to avoid ANSI escape code clutter
# when multiple agents run in parallel.
os.environ.setdefault("TQDM_DISABLE", "1")

import pandas as pd
import akshare as ak

from src.data.models import (
    Price,
    FinancialMetrics,
    LineItem,
    InsiderTrade,
    CompanyNews,
)
from src.data.line_item_mapping import (
    INCOME_STATEMENT_MAP,
    BALANCE_SHEET_MAP,
    CASH_FLOW_MAP,
    ENGLISH_TO_CHINESE,
    INCOME_ITEMS,
    BALANCE_ITEMS,
    CASHFLOW_ITEMS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe rate limiter + result cache
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 1.0  # seconds between AKShare calls (increased from 0.3)

# In-memory cache keyed by (function_name, args) to avoid redundant fetches
# when multiple agents request the same data in parallel.
_cache: dict[tuple, object] = {}


def _rate_limit():
    """Thread-safe rate limiter to avoid IP bans from scraping-based APIs."""
    global _last_call_time
    with _lock:
        elapsed = time.time() - _last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_time = time.time()


def _cache_get(key: tuple):
    """Thread-safe cache lookup."""
    with _lock:
        return _cache.get(key)


def _cache_set(key: tuple, value):
    """Thread-safe cache write."""
    with _lock:
        _cache[key] = value


def _normalize_ticker(ticker: str) -> str:
    """Strip market prefix, return pure 6-digit code."""
    return re.sub(r'^[Ss][Hh]\d*\.?|^sz\d*\.?|^SH\.?|^SZ\.?', '', str(ticker)).strip()


def _safe_float(val) -> float | None:
    """Convert value to float, return None on failure."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convert value to int, return None on failure."""
    f = _safe_float(val)
    if f is None:
        return None
    try:
        return int(f)
    except (ValueError, TypeError):
        return None


def _safe_pct(val) -> float | None:
    """Convert percentage value (e.g., 15.3 meaning 15.3%) to ratio (0.153)."""
    f = _safe_float(val)
    if f is None:
        return None
    return f / 100.0


def _fmt_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD for AKShare."""
    return date_str.replace("-", "")


# ---------------------------------------------------------------------------
# Public API functions (same signatures as src/tools/api.py)
# ---------------------------------------------------------------------------


def get_prices(ticker: str, start_date: str, end_date: str, api_key=None) -> list[Price]:
    """Fetch daily OHLCV price bars for an A-share ticker (cached)."""
    cache_key = ("get_prices", ticker, start_date, end_date)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    code = _normalize_ticker(ticker)
    df = None

    for attempt in range(3):
        _rate_limit()
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=_fmt_date(start_date),
                end_date=_fmt_date(end_date),
                adjust="qfq",
            )
            break
        except Exception as e:
            logger.warning("AKShare get_prices attempt %d failed for %s: %s", attempt + 1, ticker, e)
            if attempt < 2:
                time.sleep(3 * (attempt + 1))  # 3s, 6s
            else:
                # Fallback: try stock_zh_a_daily from Tencent
                try:
                    _rate_limit()
                    df = ak.stock_zh_a_daily(symbol=f"sz{code}" if code.startswith(('0','3')) else f"sh{code}",
                                              start_date=_fmt_date(start_date), end_date=_fmt_date(end_date), adjust="qfq")
                    break
                except Exception as e2:
                    logger.warning("AKShare get_prices fallback also failed for %s: %s", ticker, e2)
                    _cache_set(cache_key, [])
                    return []

    if df is None or df.empty:
        return []

    prices = []
    for _, row in df.iterrows():
        # Support both stock_zh_a_hist column names (开盘/收盘) and
        # stock_zh_a_daily column names (open/close)
        open_val = _safe_float(row.get("开盘") or row.get("open"))
        close_val = _safe_float(row.get("收盘") or row.get("close"))
        high_val = _safe_float(row.get("最高") or row.get("high"))
        low_val = _safe_float(row.get("最低") or row.get("low"))
        vol = _safe_int(row.get("成交量") or row.get("volume"))
        date_val = str(row.get("日期") or row.get("date", ""))[:10]

        # Skip rows with missing price data
        if open_val is None or close_val is None:
            continue

        prices.append(Price(
            open=open_val,
            close=close_val,
            high=high_val or close_val,
            low=low_val or close_val,
            volume=vol,
            time=date_val,
        ))
    _cache_set(cache_key, prices)
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key=None,
) -> list[FinancialMetrics]:
    """Fetch pre-computed financial ratios for an A-share ticker (cached)."""
    cache_key = ("get_financial_metrics", ticker, end_date, period, limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    code = _normalize_ticker(ticker)

    # Fetch raw financial statements
    income_df = _fetch_income_statement(code)
    balance_df = _fetch_balance_sheet(code)
    cashflow_df = _fetch_cashflow_statement(code)

    if income_df is None and balance_df is None:
        return []

    # Build report periods (sorted desc)
    periods = _get_report_periods(income_df, balance_df, cashflow_df)

    metrics_list = []
    for rp in periods[:limit]:
        if rp > end_date.replace("-", ""):
            continue

        inc = _get_row_by_period(income_df, rp)
        bal = _get_row_by_period(balance_df, rp)
        cff = _get_row_by_period(cashflow_df, rp)

        # Extract key values
        revenue = _sf(inc, "营业收入")
        net_income = _sf(inc, "净利润")
        operating_income = _sf(inc, "营业利润")
        gross_profit = revenue - _sf(inc, "营业成本") if revenue and _sf(inc, "营业成本") else None
        eps = _sf(inc, "基本每股收益")
        interest_expense = _sf(inc, "利息费用")

        total_assets = _sf(bal, "资产总计")
        total_liabilities = _sf(bal, "负债合计")
        current_assets = _sf(bal, "流动资产合计")
        current_liabilities = _sf(bal, "流动负债合计")
        shareholders_equity = _sf(bal, "所有者权益(或股东权益)合计")
        cash = _sf(bal, "货币资金")
        inventory = _sf(bal, "存货")
        outstanding_shares_val = _sf(bal, "实收资本(或股本)")
        short_borrowing = _sf(bal, "短期借款")
        long_borrowing = _sf(bal, "长期借款")

        operating_cash_flow = _sf(cff, "经营活动产生的现金流量净额")
        capex = _sf(cff, "购建固定资产、无形资产和其他长期资产所支付的现金")
        free_cash_flow = None
        if operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow - abs(capex)

        # Compute ratios
        gross_margin = gross_profit / revenue if gross_profit and revenue and revenue != 0 else None
        net_margin = net_income / revenue if net_income and revenue and revenue != 0 else None
        operating_margin = operating_income / revenue if operating_income and revenue and revenue != 0 else None
        roe = net_income / shareholders_equity if net_income and shareholders_equity and shareholders_equity != 0 else None
        roa = net_income / total_assets if net_income and total_assets and total_assets != 0 else None
        current_ratio = current_assets / current_liabilities if current_assets and current_liabilities and current_liabilities != 0 else None
        debt_to_equity = total_liabilities / shareholders_equity if total_liabilities and shareholders_equity and shareholders_equity != 0 else None
        debt_to_assets = total_liabilities / total_assets if total_liabilities and total_assets and total_assets != 0 else None
        interest_coverage = operating_income / interest_expense if operating_income and interest_expense and interest_expense != 0 else None
        fcf_per_share = free_cash_flow / outstanding_shares_val if free_cash_flow and outstanding_shares_val and outstanding_shares_val != 0 else None
        book_value_per_share = shareholders_equity / outstanding_shares_val if shareholders_equity and outstanding_shares_val and outstanding_shares_val != 0 else None

        total_debt = None
        if short_borrowing is not None or long_borrowing is not None:
            total_debt = (short_borrowing or 0) + (long_borrowing or 0)

        # Market cap - will be filled by get_market_cap separately
        market_cap = None

        report_period_fmt = f"{rp[:4]}-{rp[4:6]}-{rp[6:8]}"

        fm = FinancialMetrics(
            ticker=ticker,
            report_period=report_period_fmt,
            period=period,
            currency="CNY",
            market_cap=market_cap,
            enterprise_value=None,
            price_to_earnings_ratio=None,
            price_to_book_ratio=None,
            price_to_sales_ratio=None,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=current_ratio,
            quick_ratio=None,
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=debt_to_equity,
            debt_to_assets=debt_to_assets,
            interest_coverage=interest_coverage,
            revenue_growth=None,
            earnings_growth=None,
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=None,
            earnings_per_share=eps,
            book_value_per_share=book_value_per_share,
            free_cash_flow_per_share=fcf_per_share,
        )
        metrics_list.append(fm)

    # Compute growth metrics (need at least 2 periods)
    if len(metrics_list) >= 2:
        for i in range(len(metrics_list) - 1):
            curr = metrics_list[i]
            prev = metrics_list[i + 1]
            curr.revenue_growth = _growth_rate(
                _revenue_by_period(income_df, curr.report_period.replace("-", "")),
                _revenue_by_period(income_df, prev.report_period.replace("-", "")),
            )
            curr.earnings_growth = _growth_rate(
                _net_income_by_period(income_df, curr.report_period.replace("-", "")),
                _net_income_by_period(income_df, prev.report_period.replace("-", "")),
            )

    _cache_set(cache_key, metrics_list)
    return metrics_list


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key=None,
) -> list[LineItem]:
    """Fetch specific financial statement line items for an A-share ticker."""
    code = _normalize_ticker(ticker)

    # Determine which statements we need
    need_income = any(item in INCOME_ITEMS for item in line_items)
    need_balance = any(item in BALANCE_ITEMS for item in line_items)
    need_cashflow = any(item in CASHFLOW_ITEMS for item in line_items)

    # If unsure, fetch all
    if not need_income and not need_balance and not need_cashflow:
        need_income = need_balance = need_cashflow = True

    income_df = _fetch_income_statement(code) if need_income else None
    balance_df = _fetch_balance_sheet(code) if need_balance else None
    cashflow_df = _fetch_cashflow_statement(code) if need_cashflow else None

    periods = _get_report_periods(income_df, balance_df, cashflow_df)

    results = []
    for rp in periods[:limit]:
        if rp > end_date.replace("-", ""):
            continue

        inc = _get_row_by_period(income_df, rp)
        bal = _get_row_by_period(balance_df, rp)
        cff = _get_row_by_period(cashflow_df, rp)

        item_data = {
            "ticker": ticker,
            "report_period": f"{rp[:4]}-{rp[4:6]}-{rp[6:8]}",
            "period": period,
            "currency": "CNY",
        }

        # Merge all Chinese->English maps
        all_maps = {}
        all_maps.update(INCOME_STATEMENT_MAP)
        all_maps.update(BALANCE_SHEET_MAP)
        all_maps.update(CASH_FLOW_MAP)

        # Build reverse map
        cn_to_en = {cn: en for en, cn in ENGLISH_TO_CHINESE.items()}

        for english_name in line_items:
            chinese_name = ENGLISH_TO_CHINESE.get(english_name)
            if chinese_name:
                val = None
                if inc is not None and chinese_name in inc:
                    val = _safe_float(inc[chinese_name])
                elif bal is not None and chinese_name in bal:
                    val = _safe_float(bal[chinese_name])
                elif cff is not None and chinese_name in cff:
                    val = _safe_float(cff[chinese_name])
                item_data[english_name] = val
            else:
                # Try direct lookup in any statement
                item_data[english_name] = _find_in_statements(english_name, inc, bal, cff)

        # Special computed fields
        if "gross_profit" in line_items and "gross_profit" not in item_data:
            rev = _safe_float(item_data.get("revenue"))
            cost = _safe_float(item_data.get("cost_of_revenue"))
            if inc is not None:
                cost = _sf(inc, "营业成本")
            if rev and cost:
                item_data["gross_profit"] = rev - cost

        if "working_capital" in line_items and "working_capital" not in item_data:
            ca = _sf(bal, "流动资产合计")
            cl = _sf(bal, "流动负债合计")
            if ca is not None and cl is not None:
                item_data["working_capital"] = ca - cl

        if "total_debt" in line_items and "total_debt" not in item_data:
            short = _sf(bal, "短期借款") or 0
            long_ = _sf(bal, "长期借款") or 0
            item_data["total_debt"] = short + long_

        if "free_cash_flow" in line_items and "free_cash_flow" not in item_data:
            ocf = _sf(cff, "经营活动产生的现金流量净额")
            capex_ = _sf(cff, "购建固定资产、无形资产和其他长期资产所支付的现金")
            if ocf is not None and capex_ is not None:
                item_data["free_cash_flow"] = ocf - abs(capex_)

        if "ebitda" in line_items and "ebitda" not in item_data:
            op_inc = _sf(inc, "营业利润")
            # EBITDA approximation: operating_income + depreciation
            # We don't have depreciation directly, approximate from cashflow
            item_data["ebitda"] = op_inc

        if "ebit" in line_items and "ebit" not in item_data:
            item_data["ebit"] = _sf(inc, "营业利润")

        if "outstanding_shares" in line_items and "outstanding_shares" not in item_data:
            item_data["outstanding_shares"] = _sf(bal, "实收资本(或股本)")

        if "book_value_per_share" in line_items and "book_value_per_share" not in item_data:
            se = _sf(bal, "所有者权益(或股东权益)合计")
            shares = _sf(bal, "实收资本(或股本)")
            if se and shares and shares != 0:
                item_data["book_value_per_share"] = se / shares

        results.append(LineItem(**item_data))

    return results


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key=None,
) -> list[InsiderTrade]:
    """Fetch insider (高管) transaction records for an A-share ticker.

    Note: A-share insider trade data from AKShare is limited.
    Uses stock_zh_a_gdhs (股东户数) as a proxy for shareholder activity.
    """
    code = _normalize_ticker(ticker)
    _rate_limit()

    try:
        # Try 股东户数变化 as a proxy
        df = ak.stock_zh_a_gdhs(symbol=code)
        if df is None or df.empty:
            return []
    except Exception as e:
        logger.warning("AKShare get_insider_trades failed for %s: %s", ticker, e)
        return []

    trades = []
    for _, row in df.iterrows():
        date_str = str(row.get("股东户数日期", ""))[:10]
        if not date_str:
            continue
        if start_date and date_str < start_date:
            continue
        if date_str > end_date:
            continue

        holder_count = _safe_float(row.get("股东户数"))
        trades.append(InsiderTrade(
            ticker=ticker,
            issuer=ticker,
            name=None,
            title=None,
            is_board_director=None,
            transaction_date=date_str,
            transaction_shares=holder_count,  # proxy: shareholder count change
            transaction_price_per_share=None,
            transaction_value=None,
            shares_owned_before_transaction=None,
            shares_owned_after_transaction=None,
            security_title=None,
            filing_date=date_str,
        ))

    return trades[:limit]


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key=None,
) -> list[CompanyNews]:
    """Fetch news articles for an A-share ticker from East Money."""
    code = _normalize_ticker(ticker)
    _rate_limit()

    try:
        df = ak.stock_news_em(symbol=code)
    except Exception as e:
        logger.warning("AKShare get_company_news failed for %s: %s", ticker, e)
        return []

    if df is None or df.empty:
        return []

    news_list = []
    for _, row in df.iterrows():
        date_str = str(row.get("发布时间", ""))[:10]
        if not date_str:
            continue
        if start_date and date_str < start_date:
            continue
        if date_str > end_date:
            continue

        news_list.append(CompanyNews(
            ticker=ticker,
            title=str(row.get("新闻标题", "")),
            author=None,
            source=str(row.get("文章来源", "")),
            date=date_str,
            url=str(row.get("新闻链接", "")),
            sentiment=None,  # LLM will classify later
        ))

    return news_list[:limit]


def get_market_cap(ticker: str, end_date: str, api_key=None) -> float | None:
    """Get current market capitalization for an A-share ticker (cached)."""
    cache_key = ("get_market_cap", ticker, end_date)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    code = _normalize_ticker(ticker)

    # For today: use real-time spot data
    if end_date == datetime.now().strftime("%Y-%m-%d"):
        _rate_limit()
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and not df.empty:
                # Look for 总市值 row
                for _, row in df.iterrows():
                    if "总市值" in str(row.iloc[0]):
                        val = _safe_float(row.iloc[1])
                        _cache_set(cache_key, val)
                        return val
        except Exception as e:
            logger.warning("AKShare get_market_cap (spot) failed for %s: %s", ticker, e)

    # For historical dates: price * shares outstanding
    prices = get_prices(ticker, end_date, end_date)
    if not prices:
        # Try a range around end_date
        from datetime import timedelta
        d = datetime.strptime(end_date, "%Y-%m-%d")
        for delta in range(1, 10):
            new_end = (d - timedelta(days=delta)).strftime("%Y-%m-%d")
            prices = get_prices(ticker, new_end, new_end)
            if prices:
                break

    if not prices:
        return None

    latest_price = prices[-1].close

    # Get shares outstanding from balance sheet
    balance_df = _fetch_balance_sheet(code)
    if balance_df is not None and not balance_df.empty:
        shares = _sf(balance_df.iloc[0], "实收资本(或股本)")
        if shares and latest_price:
            val = shares * latest_price
            _cache_set(cache_key, val)
            return val

    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert list of Price models to pandas DataFrame (same as original)."""
    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)

    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Cache for financial statements to avoid repeated fetches
_statement_cache: dict[str, pd.DataFrame] = {}


def _fetch_income_statement(code: str) -> pd.DataFrame | None:
    key = f"{code}_income"
    if key in _statement_cache:
        return _statement_cache[key]
    _rate_limit()
    try:
        df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
        if df is not None and not df.empty:
            # Keep only 合并期末 (consolidated) rows
            if "类型" in df.columns:
                df = df[df["类型"] == "合并期末"]
        _statement_cache[key] = df
        return df
    except Exception as e:
        logger.warning("AKShare income statement failed for %s: %s", code, e)
        return None


def _fetch_balance_sheet(code: str) -> pd.DataFrame | None:
    key = f"{code}_balance"
    if key in _statement_cache:
        return _statement_cache[key]
    _rate_limit()
    try:
        df = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
        if df is not None and not df.empty:
            if "类型" in df.columns:
                df = df[df["类型"] == "合并期末"]
        _statement_cache[key] = df
        return df
    except Exception as e:
        logger.warning("AKShare balance sheet failed for %s: %s", code, e)
        return None


def _fetch_cashflow_statement(code: str) -> pd.DataFrame | None:
    key = f"{code}_cashflow"
    if key in _statement_cache:
        return _statement_cache[key]
    _rate_limit()
    try:
        df = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
        if df is not None and not df.empty:
            if "类型" in df.columns:
                df = df[df["类型"] == "合并期末"]
        _statement_cache[key] = df
        return df
    except Exception as e:
        logger.warning("AKShare cash flow failed for %s: %s", code, e)
        return None


def _get_report_periods(*dfs) -> list[str]:
    """Extract sorted (desc) unique report periods from DataFrames."""
    periods = set()
    for df in dfs:
        if df is not None and not df.empty and "报告日" in df.columns:
            for v in df["报告日"].dropna().unique():
                s = str(v).replace("-", "").strip()[:8]
                if len(s) == 8 and s.isdigit():
                    periods.add(s)
    return sorted(periods, reverse=True)


def _get_row_by_period(df: pd.DataFrame | None, period: str) -> dict | None:
    """Get the first row matching a report period as a dict."""
    if df is None or df.empty or "报告日" not in df.columns:
        return None
    matches = df[df["报告日"].astype(str).str.replace("-", "").str[:8] == period]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _sf(row: dict | None, col: str) -> float | None:
    """Safe float from a row dict."""
    if row is None:
        return None
    return _safe_float(row.get(col))


def _find_in_statements(english_name: str, *rows) -> float | None:
    """Try to find a value by English name in any statement row."""
    chinese_name = ENGLISH_TO_CHINESE.get(english_name)
    if not chinese_name:
        return None
    for row in rows:
        if row is not None and chinese_name in row:
            return _safe_float(row[chinese_name])
    return None


def _growth_rate(curr: float | None, prev: float | None) -> float | None:
    """Compute year-over-year growth rate."""
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _revenue_by_period(df: pd.DataFrame | None, period: str) -> float | None:
    row = _get_row_by_period(df, period)
    return _sf(row, "营业收入")


def _net_income_by_period(df: pd.DataFrame | None, period: str) -> float | None:
    row = _get_row_by_period(df, period)
    return _sf(row, "净利润")
