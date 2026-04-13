from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import math
from datetime import datetime, timedelta
from typing_extensions import Literal
import numpy as np
import pandas as pd

from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class NassimTalebSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


def nassim_taleb_agent(state: AgentState, agent_id: str = "nassim_taleb_agent"):
    """Analyzes stocks using Taleb's antifragility, tail risk, and convexity principles."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    # Look one year back for insider trades and news
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data = {}
    taleb_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取价格数据")
        prices = get_prices(ticker, start_date, end_date, api_key=api_key)
        prices_df = prices_to_df(prices) if prices else pd.DataFrame()

        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取财务科目")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "revenue",
                "operating_income",
                "research_and_development",
                "capital_expenditure",
                "outstanding_shares",
            ],
            end_date,
            period="ttm",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "获取内部人交易")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status(agent_id, ticker, "获取公司新闻")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=100)

        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # Run sub-analyses
        progress.update_status(agent_id, ticker, "分析尾部风险")
        tail_risk_analysis = analyze_tail_risk(prices_df)

        progress.update_status(agent_id, ticker, "分析反脆弱性")
        antifragility_analysis = analyze_antifragility(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "分析凸性")
        convexity_analysis = analyze_convexity(metrics, line_items, prices_df, market_cap)

        progress.update_status(agent_id, ticker, "分析脆弱性")
        fragility_analysis = analyze_fragility(metrics, line_items)

        progress.update_status(agent_id, ticker, "分析利益一致性")
        skin_in_game_analysis = analyze_skin_in_game(insider_trades)

        progress.update_status(agent_id, ticker, "分析波动率体制")
        volatility_regime_analysis = analyze_volatility_regime(prices_df)

        progress.update_status(agent_id, ticker, "扫描黑天鹅信号")
        black_swan_analysis = analyze_black_swan_sentinel(news, prices_df)

        # Aggregate scores (raw addition — max_scores create implicit weighting)
        total_score = (
            tail_risk_analysis["score"]
            + antifragility_analysis["score"]
            + convexity_analysis["score"]
            + fragility_analysis["score"]
            + skin_in_game_analysis["score"]
            + volatility_regime_analysis["score"]
            + black_swan_analysis["score"]
        )
        max_possible_score = (
            tail_risk_analysis["max_score"]
            + antifragility_analysis["max_score"]
            + convexity_analysis["max_score"]
            + fragility_analysis["max_score"]
            + skin_in_game_analysis["max_score"]
            + volatility_regime_analysis["max_score"]
            + black_swan_analysis["max_score"]
        )

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "tail_risk_analysis": tail_risk_analysis,
            "antifragility_analysis": antifragility_analysis,
            "convexity_analysis": convexity_analysis,
            "fragility_analysis": fragility_analysis,
            "skin_in_game_analysis": skin_in_game_analysis,
            "volatility_regime_analysis": volatility_regime_analysis,
            "black_swan_analysis": black_swan_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "生成塔勒布分析")
        taleb_output = generate_taleb_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        taleb_analysis[ticker] = {
            "signal": taleb_output.signal,
            "confidence": taleb_output.confidence,
            "reasoning": taleb_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=taleb_output.reasoning)

    # Create the message
    message = HumanMessage(content=json.dumps(taleb_analysis), name=agent_id)

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(taleb_analysis, "纳西姆·塔勒布智能体")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = taleb_analysis

    progress.update_status(agent_id, None, "完成")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Helper
###############################################################################


def safe_float(value, default=0.0):
    """Safely convert a value to float, handling NaN cases."""
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


###############################################################################
# Sub-analysis functions
###############################################################################


def analyze_tail_risk(prices_df: pd.DataFrame) -> dict[str, any]:
    """Assess fat tails, skewness, tail ratio, and max drawdown."""
    if prices_df.empty or len(prices_df) < 20:
        return {"score": 0, "max_score": 8, "details": "尾部风险分析价格数据不足"}

    score = 0
    reasoning = []

    returns = prices_df["close"].pct_change().dropna()

    # Excess kurtosis (use rolling 63-day if enough data, else full series)
    if len(returns) >= 63:
        kurt = safe_float(returns.rolling(63).kurt().iloc[-1])
    else:
        kurt = safe_float(returns.kurt())

    if kurt > 5:
        score += 2
        reasoning.append(f"极度肥尾（峰度 {kurt:.1f}）")
    elif kurt > 2:
        score += 1
        reasoning.append(f"中等肥尾（峰度 {kurt:.1f}）")
    else:
        reasoning.append(f"近似正态尾部（峰度 {kurt:.1f}）——可疑的薄尾")

    # Skewness
    if len(returns) >= 63:
        skew = safe_float(returns.rolling(63).skew().iloc[-1])
    else:
        skew = safe_float(returns.skew())

    if skew > 0.5:
        score += 2
        reasoning.append(f"正偏度（{skew:.2f}）有利于做多凸性")
    elif skew > -0.5:
        score += 1
        reasoning.append(f"对称分布（偏度 {skew:.2f}）")
    else:
        reasoning.append(f"负偏度（{skew:.2f}）——易崩盘")

    # Tail ratio (95th percentile gains / abs(5th percentile losses))
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    if len(positive_returns) > 20 and len(negative_returns) > 20:
        right_tail = np.percentile(positive_returns, 95)
        left_tail = abs(np.percentile(negative_returns, 5))
        tail_ratio = right_tail / left_tail if left_tail > 0 else 1.0

        if tail_ratio > 1.2:
            score += 2
            reasoning.append(f"非对称上行（尾部比 {tail_ratio:.2f}）")
        elif tail_ratio > 0.8:
            score += 1
            reasoning.append(f"均衡尾部（尾部比 {tail_ratio:.2f}）")
        else:
            reasoning.append(f"非对称下行（尾部比 {tail_ratio:.2f}）")
    else:
        reasoning.append("尾部比数据不足")

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = safe_float(drawdown.min())

    if max_dd > -0.15:
        score += 2
        reasoning.append(f"抗跌性强（最大回撤 {max_dd:.1%}）")
    elif max_dd > -0.30:
        score += 1
        reasoning.append(f"中等回撤（{max_dd:.1%}）")
    else:
        reasoning.append(f"严重回撤（{max_dd:.1%}）——脆弱")

    return {"score": score, "max_score": 8, "details": "; ".join(reasoning)}


def analyze_antifragility(metrics: list, line_items: list, market_cap: float | None) -> dict[str, any]:
    """Evaluate whether the company benefits from disorder: low debt, high cash, stable margins."""
    if not metrics and not line_items:
        return {"score": 0, "max_score": 10, "details": "反脆弱性分析数据不足"}

    score = 0
    reasoning = []
    latest_metrics = metrics[0] if metrics else None
    latest_item = line_items[0] if line_items else None

    # Net cash position
    cash = getattr(latest_item, "cash_and_equivalents", None) if latest_item else None
    total_debt = getattr(latest_item, "total_debt", None) if latest_item else None
    total_assets = getattr(latest_item, "total_assets", None) if latest_item else None

    if cash is not None and total_debt is not None:
        net_cash = cash - total_debt
        if net_cash > 0 and market_cap and cash > 0.20 * market_cap:
            score += 3
            reasoning.append(f"现金弹药库：净现金 ${net_cash:,.0f}，现金占市值 {cash / market_cap:.0%}")
        elif net_cash > 0:
            score += 2
            reasoning.append(f"净现金为正（${net_cash:,.0f}）")
        elif total_assets and total_debt < 0.30 * total_assets:
            score += 1
            reasoning.append("净负债但相对于资产可控")
        else:
            reasoning.append("杠杆头寸——不具备反脆弱性")
    else:
        reasoning.append("现金/负债数据不可用")

    # Debt-to-equity
    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.3:
            score += 2
            reasoning.append(f"塔勒布认可的低杠杆（D/E {debt_to_equity:.2f}）")
        elif debt_to_equity < 0.7:
            score += 1
            reasoning.append(f"中等杠杆（D/E {debt_to_equity:.2f}）")
        else:
            reasoning.append(f"高杠杆（D/E {debt_to_equity:.2f}）——脆弱")
    else:
        reasoning.append("负债权益比数据不可用")

    # Operating margin stability (CV across periods)
    op_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(op_margins) >= 3:
        mean_margin = sum(op_margins) / len(op_margins)
        variance = sum((m - mean_margin) ** 2 for m in op_margins) / len(op_margins)
        std_margin = variance ** 0.5
        cv = std_margin / abs(mean_margin) if mean_margin != 0 else float("inf")

        if cv < 0.15 and mean_margin > 0.15:
            score += 3
            reasoning.append(f"稳定的高利润率（均值 {mean_margin:.1%}，变异系数 {cv:.2f}）——反脆弱的定价权")
        elif cv < 0.30 and mean_margin > 0.10:
            score += 2
            reasoning.append(f"利润率稳定性尚可（均值 {mean_margin:.1%}，变异系数 {cv:.2f}）")
        elif cv < 0.30:
            score += 1
            reasoning.append(f"利润率相对稳定（变异系数 {cv:.2f}）但偏低（均值 {mean_margin:.1%}）")
        else:
            reasoning.append(f"利润率波动大（变异系数 {cv:.2f}）——脆弱的定价权")
    else:
        reasoning.append("利润率历史数据不足以进行稳定性分析")

    # FCF consistency
    fcf_values = [getattr(item, "free_cash_flow", None) for item in line_items] if line_items else []
    fcf_values = [v for v in fcf_values if v is not None]
    if fcf_values:
        positive_count = sum(1 for v in fcf_values if v > 0)
        if positive_count == len(fcf_values):
            score += 2
            reasoning.append(f"持续的自由现金流产生（{positive_count}/{len(fcf_values)} 个期间为正）")
        elif positive_count > len(fcf_values) / 2:
            score += 1
            reasoning.append(f"多数期间自由现金流为正（{positive_count}/{len(fcf_values)} 个期间）")
        else:
            reasoning.append(f"自由现金流不稳定（{positive_count}/{len(fcf_values)} 个期间为正）")
    else:
        reasoning.append("自由现金流数据不可用")

    return {"score": score, "max_score": 10, "details": "; ".join(reasoning)}


def analyze_convexity(
    metrics: list, line_items: list, prices_df: pd.DataFrame, market_cap: float | None
) -> dict[str, any]:
    """Measure asymmetric payoff potential: R&D optionality, upside/downside ratio, cash optionality."""
    if not metrics and not line_items and prices_df.empty:
        return {"score": 0, "max_score": 10, "details": "凸性分析数据不足"}

    score = 0
    reasoning = []
    latest_item = line_items[0] if line_items else None

    # R&D as embedded optionality
    rd = getattr(latest_item, "research_and_development", None) if latest_item else None
    revenue = getattr(latest_item, "revenue", None) if latest_item else None

    if rd is not None and revenue and revenue > 0:
        rd_ratio = abs(rd) / revenue
        if rd_ratio > 0.15:
            score += 3
            reasoning.append(f"通过研发具有显著的嵌入式期权价值（占收入 {rd_ratio:.1%}）")
        elif rd_ratio > 0.08:
            score += 2
            reasoning.append(f"有意义的研发投入（占收入 {rd_ratio:.1%}）")
        elif rd_ratio > 0.03:
            score += 1
            reasoning.append(f"适度的研发（占收入 {rd_ratio:.1%}）")
        else:
            reasoning.append(f"极少研发（占收入 {rd_ratio:.1%}）")
    else:
        reasoning.append("研发数据不可用——非研发行业不扣分")

    # Upside/downside capture ratio
    if not prices_df.empty and len(prices_df) >= 20:
        returns = prices_df["close"].pct_change().dropna()
        upside = returns[returns > 0]
        downside = returns[returns < 0]

        if len(upside) > 10 and len(downside) > 10:
            avg_up = upside.mean()
            avg_down = abs(downside.mean())
            up_down_ratio = avg_up / avg_down if avg_down > 0 else 1.0

            if up_down_ratio > 1.3:
                score += 2
                reasoning.append(f"凸性收益特征（上行/下行比 {up_down_ratio:.2f}）")
            elif up_down_ratio > 1.0:
                score += 1
                reasoning.append(f"轻微正不对称（上行/下行比 {up_down_ratio:.2f}）")
            else:
                reasoning.append(f"凹性收益（上行/下行比 {up_down_ratio:.2f}）——不利")
        else:
            reasoning.append("收益率数据不足以进行不对称分析")
    else:
        reasoning.append("价格数据不足以进行收益率不对称分析")

    # Cash optionality (cash / market_cap)
    cash = getattr(latest_item, "cash_and_equivalents", None) if latest_item else None
    if cash is not None and market_cap and market_cap > 0:
        cash_ratio = cash / market_cap
        if cash_ratio > 0.30:
            score += 3
            reasoning.append(f"现金是未来机会的看涨期权（占市值 {cash_ratio:.0%}）")
        elif cash_ratio > 0.15:
            score += 2
            reasoning.append(f"强劲的现金头寸（占市值 {cash_ratio:.0%}）")
        elif cash_ratio > 0.05:
            score += 1
            reasoning.append(f"适度的现金缓冲（占市值 {cash_ratio:.0%}）")
        else:
            reasoning.append(f"现金相对市值较低（{cash_ratio:.0%}）")
    else:
        reasoning.append("现金/市值数据不可用")

    # FCF yield
    latest_metrics = metrics[0] if metrics else None
    fcf_yield = None
    if latest_item and market_cap and market_cap > 0:
        fcf = getattr(latest_item, "free_cash_flow", None)
        if fcf is not None:
            fcf_yield = fcf / market_cap
    if fcf_yield is None and latest_metrics:
        fcf_yield = getattr(latest_metrics, "free_cash_flow_yield", None)

    if fcf_yield is not None:
        if fcf_yield > 0.10:
            score += 2
            reasoning.append(f"高自由现金流收益率（{fcf_yield:.1%}）为凸性投资提供安全边际")
        elif fcf_yield > 0.05:
            score += 1
            reasoning.append(f"尚可的自由现金流收益率（{fcf_yield:.1%}）")
        else:
            reasoning.append(f"低自由现金流收益率（{fcf_yield:.1%}）")
    else:
        reasoning.append("自由现金流收益率数据不可用")

    return {"score": score, "max_score": 10, "details": "; ".join(reasoning)}


def analyze_fragility(metrics: list, line_items: list) -> dict[str, any]:
    """Via Negativa: detect fragile companies. High score = NOT fragile."""
    if not metrics:
        return {"score": 0, "max_score": 8, "details": "脆弱性分析数据不足"}

    score = 0
    reasoning = []
    latest_metrics = metrics[0]

    # Leverage fragility
    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None)
    if debt_to_equity is not None:
        if debt_to_equity > 2.0:
            reasoning.append(f"极度脆弱的资产负债表（D/E {debt_to_equity:.2f}）")
        elif debt_to_equity > 1.0:
            score += 1
            reasoning.append(f"杠杆偏高（D/E {debt_to_equity:.2f}）")
        elif debt_to_equity > 0.5:
            score += 2
            reasoning.append(f"中等杠杆（D/E {debt_to_equity:.2f}）")
        else:
            score += 3
            reasoning.append(f"低杠杆（D/E {debt_to_equity:.2f}）——不脆弱")
    else:
        reasoning.append("负债权益比数据不可用")

    # Interest coverage
    interest_coverage = getattr(latest_metrics, "interest_coverage", None)
    if interest_coverage is not None:
        if interest_coverage > 10:
            score += 2
            reasoning.append(f"利息覆盖率 {interest_coverage:.1f} 倍——债务无关紧要")
        elif interest_coverage > 5:
            score += 1
            reasoning.append(f"利息覆盖率充裕（{interest_coverage:.1f} 倍）")
        else:
            reasoning.append(f"利息覆盖率低（{interest_coverage:.1f} 倍）——对利率变化脆弱")
    else:
        reasoning.append("利息覆盖率数据不可用")

    # Earnings volatility
    earnings_growth_values = [m.earnings_growth for m in metrics if m.earnings_growth is not None]
    if len(earnings_growth_values) >= 3:
        mean_eg = sum(earnings_growth_values) / len(earnings_growth_values)
        variance = sum((e - mean_eg) ** 2 for e in earnings_growth_values) / len(earnings_growth_values)
        std_eg = variance ** 0.5

        if std_eg < 0.20:
            score += 2
            reasoning.append(f"盈利稳定（增长标准差 {std_eg:.2f}）——稳健")
        elif std_eg < 0.50:
            score += 1
            reasoning.append(f"中等盈利波动（增长标准差 {std_eg:.2f}）")
        else:
            reasoning.append(f"盈利高度波动（增长标准差 {std_eg:.2f}）——脆弱")
    else:
        reasoning.append("盈利历史不足以进行波动率分析")

    # Net margin buffer
    net_margin = getattr(latest_metrics, "net_margin", None)
    if net_margin is not None:
        if net_margin > 0.15:
            score += 1
            reasoning.append(f"丰厚的利润率（{net_margin:.1%}）可缓冲冲击")
        elif net_margin >= 0.05:
            reasoning.append(f"中等的利润率（{net_margin:.1%}）")
        else:
            reasoning.append(f"微薄的利润率（{net_margin:.1%}）——一次冲击就可能亏损")
    else:
        reasoning.append("净利润率数据不可用")

    # Clamp score at minimum 0
    score = max(score, 0)

    return {"score": score, "max_score": 8, "details": "; ".join(reasoning)}


def analyze_skin_in_game(insider_trades: list) -> dict[str, any]:
    """Assess insider alignment: net insider buying signals trust."""
    if not insider_trades:
        return {"score": 1, "max_score": 4, "details": "无内部人交易数据——中性假设"}

    score = 0
    reasoning = []

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold

    if net > 0:
        buy_sell_ratio = net / max(shares_sold, 1)
        if buy_sell_ratio > 2.0:
            score = 4
            reasoning.append(f"强烈的利益一致性——内部人净买入 {net:,} 股（比率 {buy_sell_ratio:.1f} 倍）")
        elif buy_sell_ratio > 0.5:
            score = 3
            reasoning.append(f"中等内部人信心——净买入 {net:,} 股")
        else:
            score = 2
            reasoning.append(f"内部人净买入 {net:,} 股")
    else:
        reasoning.append(f"内部人在卖出——缺乏利益一致性（净 {net:,} 股）")

    return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}


def analyze_volatility_regime(prices_df: pd.DataFrame) -> dict[str, any]:
    """Volatility regime analysis. Key Taleb insight: low vol is dangerous (turkey problem)."""
    if prices_df.empty or len(prices_df) < 30:
        return {"score": 0, "max_score": 6, "details": "波动率分析价格数据不足"}

    score = 0
    reasoning = []

    returns = prices_df["close"].pct_change().dropna()

    # Historical volatility (annualized, 21-day rolling)
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Vol regime ratio (current vol / 63-day avg vol)
    if len(hist_vol.dropna()) >= 63:
        vol_ma = hist_vol.rolling(63).mean()
        current_vol = safe_float(hist_vol.iloc[-1])
        avg_vol = safe_float(vol_ma.iloc[-1])
        vol_regime = current_vol / avg_vol if avg_vol > 0 else 1.0
    elif len(hist_vol.dropna()) >= 21:
        # Fallback: compare current to overall mean
        current_vol = safe_float(hist_vol.iloc[-1])
        avg_vol = safe_float(hist_vol.mean())
        vol_regime = current_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        return {"score": 0, "max_score": 6, "details": "波动率体制分析数据不足"}

    # Vol regime scoring (max 4)
    if vol_regime < 0.7:
        reasoning.append(f"危险的低波动率（体制 {vol_regime:.2f}）——火鸡问题")
    elif vol_regime < 0.9:
        score += 1
        reasoning.append(f"低于平均的波动率（体制 {vol_regime:.2f}）——接近自满")
    elif vol_regime <= 1.3:
        score += 3
        reasoning.append(f"正常波动率体制（{vol_regime:.2f}）——合理定价")
    elif vol_regime <= 2.0:
        score += 4
        reasoning.append(f"波动率升高（体制 {vol_regime:.2f}）——反脆弱者的机会")
    else:
        score += 2
        reasoning.append(f"极端波动率（体制 {vol_regime:.2f}）——危机模式")

    # Vol-of-vol scoring (max 2)
    if len(hist_vol.dropna()) >= 42:
        vol_of_vol = hist_vol.rolling(21).std()
        vol_of_vol_clean = vol_of_vol.dropna()
        if len(vol_of_vol_clean) > 0:
            current_vov = safe_float(vol_of_vol_clean.iloc[-1])
            median_vov = safe_float(vol_of_vol_clean.median())
            if median_vov > 0:
                if current_vov > 2 * median_vov:
                    score += 2
                    reasoning.append(f"波动率高度不稳定（波动率的波动率 {current_vov:.4f} vs 中位数 {median_vov:.4f}）——体制转换可能")
                elif current_vov > median_vov:
                    score += 1
                    reasoning.append(f"波动率的波动率升高（{current_vov:.4f} vs 中位数 {median_vov:.4f}）")
                else:
                    reasoning.append(f"波动率的波动率稳定（{current_vov:.4f}）")
            else:
                reasoning.append("波动率的波动率中位数为零——异常")
        else:
            reasoning.append("波动率的波动率数据不足")
    else:
        reasoning.append("波动率的波动率分析历史数据不足")

    return {"score": score, "max_score": 6, "details": "; ".join(reasoning)}


def analyze_black_swan_sentinel(news: list, prices_df: pd.DataFrame) -> dict[str, any]:
    """Monitor for crisis signals: abnormal news sentiment, volume spikes, price dislocations."""
    score = 2  # Default: normal conditions
    reasoning = []

    # News sentiment analysis
    neg_ratio = 0.0
    if news:
        total = len(news)
        neg_count = sum(1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"])
        neg_ratio = neg_count / total if total > 0 else 0
    else:
        reasoning.append("无近期新闻数据")

    # Volume spike detection
    volume_spike = 1.0
    recent_return = 0.0
    if not prices_df.empty and len(prices_df) >= 10:
        if "volume" in prices_df.columns:
            recent_vol = prices_df["volume"].iloc[-5:].mean()
            avg_vol = prices_df["volume"].iloc[-63:].mean() if len(prices_df) >= 63 else prices_df["volume"].mean()
            volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0

        if len(prices_df) >= 5:
            recent_return = safe_float(prices_df["close"].iloc[-1] / prices_df["close"].iloc[-5] - 1)

    # Scoring
    if neg_ratio > 0.7 and volume_spike > 2.0:
        score = 0
        reasoning.append(f"黑天鹅警告——{neg_ratio:.0%} 负面新闻，成交量 {volume_spike:.1f} 倍激增")
    elif neg_ratio > 0.5 or volume_spike > 2.5:
        score = 1
        reasoning.append(f"压力信号升高（负面新闻 {neg_ratio:.0%}，成交量 {volume_spike:.1f} 倍）")
    elif neg_ratio > 0.3 and abs(recent_return) > 0.10:
        score = 1
        reasoning.append(f"中度压力伴随价格偏离（{recent_return:.1%} 变动，{neg_ratio:.0%} 负面新闻）")
    elif neg_ratio < 0.3 and volume_spike < 1.5:
        score = 3
        reasoning.append("未检测到黑天鹅信号")
    else:
        reasoning.append(f"正常状况（负面新闻 {neg_ratio:.0%}，成交量 {volume_spike:.1f} 倍）")

    # Contrarian bonus: high negative news but no volume panic could be opportunity
    if neg_ratio > 0.4 and volume_spike < 1.5 and score < 4:
        score = min(score + 1, 4)
        reasoning.append("逆向投资机会——负面情绪但无恐慌性抛售")

    return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}


###############################################################################
# LLM generation
###############################################################################


def generate_taleb_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str = "nassim_taleb_agent",
) -> NassimTalebSignal:
    """Get investment decision from LLM with a compact prompt."""

    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "tail_risk": analysis_data.get("tail_risk_analysis", {}).get("details"),
        "antifragility": analysis_data.get("antifragility_analysis", {}).get("details"),
        "convexity": analysis_data.get("convexity_analysis", {}).get("details"),
        "fragility": analysis_data.get("fragility_analysis", {}).get("details"),
        "skin_in_game": analysis_data.get("skin_in_game_analysis", {}).get("details"),
        "volatility_regime": analysis_data.get("volatility_regime_analysis", {}).get("details"),
        "black_swan": analysis_data.get("black_swan_analysis", {}).get("details"),
        "market_cap": analysis_data.get("market_cap"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是纳西姆·塔勒布。仅根据提供的事实判断看涨、看跌或中性。\n"
                "\n"
                "决策检查清单：\n"
                "- 反脆弱性（从混乱中获益）\n"
                "- 尾部风险特征（肥尾、偏度）\n"
                "- 凸性（非对称收益潜力）\n"
                "- 脆弱性否定法（避免脆弱的）\n"
                "- 利益一致性（内部人对齐）\n"
                "- 波动率体制（低波动率 = 危险）\n"
                "\n"
                "信号规则：\n"
                "- 看涨：反脆弱的业务具有凸性收益且不脆弱。\n"
                "- 看跌：脆弱的业务（高杠杆、薄利润率、盈利波动）或缺乏利益一致性。\n"
                "- 中性：信号混杂，或数据不足以判断脆弱性。\n"
                "\n"
                "置信度范围：\n"
                "- 90-100%：真正反脆弱，具有强凸性和利益一致性\n"
                "- 70-89%：低脆弱性，有不错的期权性价值\n"
                "- 50-69%：脆弱性信号混杂，尾部风险不确定\n"
                "- 30-49%：检测到部分脆弱性，内部人对齐弱\n"
                "- 10-29%：明显脆弱或危险的波动率体制\n"
                "\n"
                "使用塔勒布的词汇：反脆弱、凸性、利益一致性、否定法、杠铃策略、火鸡问题、林迪效应。\n"
                "推理保持在150字以内。不要编造数据。仅返回JSON。用中文输出推理。",
            ),
            (
                "human",
                "股票代码：{ticker}\n"
                "事实数据：\n{facts}\n\n"
                "请严格按照以下格式返回：\n"
                "{{\n"
                '  "signal": "bullish" | "bearish" | "neutral",\n'
                '  "confidence": 整数,\n'
                '  "reasoning": "用中文写的简短理由"\n'
                "}}",
            ),
        ]
    )

    prompt = template.invoke({
        "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
        "ticker": ticker,
    })

    def create_default_nassim_taleb_signal():
        return NassimTalebSignal(signal="neutral", confidence=50, reasoning="数据不足")

    return call_llm(
        prompt=prompt,
        pydantic_model=NassimTalebSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_nassim_taleb_signal,
    )
