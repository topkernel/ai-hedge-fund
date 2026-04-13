from __future__ import annotations

"""Growth Agent

Implements a growth-focused valuation methodology.
"""

import json
import statistics
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state
from src.tools.api import (
    get_financial_metrics,
    get_insider_trades,
)

def growth_analyst_agent(state: AgentState, agent_id: str = "growth_analyst_agent"):
    """Run growth analysis across tickers and write signals back to `state`."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    growth_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务数据")

        # --- Historical financial metrics ---
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=12, # 3 years of ttm data
            api_key=api_key,
        )
        if not financial_metrics or len(financial_metrics) < 4:
            progress.update_status(agent_id, ticker, "失败：财务指标不足")
            continue
        
        most_recent_metrics = financial_metrics[0]

        # --- Insider Trades ---
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key
        )

        # ------------------------------------------------------------------
        # Tool Implementation
        # ------------------------------------------------------------------
        
        # 1. Historical Growth Analysis
        growth_trends = analyze_growth_trends(financial_metrics)
        
        # 2. Growth-Oriented Valuation
        valuation_metrics = analyze_valuation(most_recent_metrics)
        
        # 3. Margin Expansion Monitor
        margin_trends = analyze_margin_trends(financial_metrics)
        
        # 4. Insider Conviction Tracker
        insider_conviction = analyze_insider_conviction(insider_trades)
        
        # 5. Financial Health Check
        financial_health = check_financial_health(most_recent_metrics)

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        scores = {
            "growth": growth_trends['score'],
            "valuation": valuation_metrics['score'],
            "margins": margin_trends['score'],
            "insider": insider_conviction['score'],
            "health": financial_health['score']
        }
        
        weights = {
            "growth": 0.40,
            "valuation": 0.25,
            "margins": 0.15,
            "insider": 0.10,
            "health": 0.10
        }

        weighted_score = sum(scores[key] * weights[key] for key in scores)
        
        if weighted_score > 0.6:
            signal = "bullish"
        elif weighted_score < 0.4:
            signal = "bearish"
        else:
            signal = "neutral"
            
        confidence = round(abs(weighted_score - 0.5) * 2 * 100)

        reasoning = {
            "historical_growth": growth_trends,
            "growth_valuation": valuation_metrics,
            "margin_expansion": margin_trends,
            "insider_conviction": insider_conviction,
            "financial_health": financial_health,
            "final_analysis": {
                "signal": signal,
                "confidence": confidence,
                "weighted_score": round(weighted_score, 2)
            }
        }

        reasoning_text = _build_reasoning_text(reasoning)
        growth_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning_text,
            "reasoning_data": reasoning,
        }
        progress.update_status(agent_id, ticker, "完成", analysis=reasoning_text)

    # ---- Emit message (for LLM tool chain) ----
    msg = HumanMessage(content=json.dumps(growth_analysis), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(growth_analysis, "增长分析师")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = growth_analysis

    progress.update_status(agent_id, None, "完成")
    
    return {"messages": [msg], "data": data}

#############################
# Helper Functions
#############################

def _calculate_trend(data: list[float | None]) -> float:
    """Calculates the slope of the trend line for the given data."""
    clean_data = [d for d in data if d is not None]
    if len(clean_data) < 2:
        return 0.0
    
    y = clean_data
    x = list(range(len(y)))
    
    try:
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(i * j for i, j in zip(x, y))
        sum_x2 = sum(i**2 for i in x)
        n = len(y)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        return slope
    except ZeroDivisionError:
        return 0.0

def analyze_growth_trends(metrics: list) -> dict:
    """Analyzes historical growth trends."""
    
    rev_growth = [m.revenue_growth for m in metrics]
    eps_growth = [m.earnings_per_share_growth for m in metrics]
    fcf_growth = [m.free_cash_flow_growth for m in metrics]

    rev_trend = _calculate_trend(rev_growth)
    eps_trend = _calculate_trend(eps_growth)
    fcf_trend = _calculate_trend(fcf_growth)

    # Score based on recent growth and trend
    score = 0
    
    # Revenue
    if rev_growth[0] is not None:
        if rev_growth[0] > 0.20:
            score += 0.4
        elif rev_growth[0] > 0.10:
            score += 0.2
        if rev_trend > 0:
            score += 0.1 # Accelerating
            
    # EPS
    if eps_growth[0] is not None:
        if eps_growth[0] > 0.20:
            score += 0.25
        elif eps_growth[0] > 0.10:
            score += 0.1
        if eps_trend > 0:
            score += 0.05
    
    # FCF
    if fcf_growth[0] is not None:
        if fcf_growth[0] > 0.15:
            score += 0.1
            
    score = min(score, 1.0)

    return {
        "score": score,
        "revenue_growth": rev_growth[0],
        "revenue_trend": rev_trend,
        "eps_growth": eps_growth[0],
        "eps_trend": eps_trend,
        "fcf_growth": fcf_growth[0],
        "fcf_trend": fcf_trend
    }

def analyze_valuation(metrics) -> dict:
    """Analyzes valuation from a growth perspective."""
    
    peg_ratio = metrics.peg_ratio
    ps_ratio = metrics.price_to_sales_ratio
    
    score = 0
    
    # PEG Ratio
    if peg_ratio is not None:
        if peg_ratio < 1.0:
            score += 0.5
        elif peg_ratio < 2.0:
            score += 0.25
            
    # Price to Sales Ratio
    if ps_ratio is not None:
        if ps_ratio < 2.0:
            score += 0.5
        elif ps_ratio < 5.0:
            score += 0.25
            
    score = min(score, 1.0)
    
    return {
        "score": score,
        "peg_ratio": peg_ratio,
        "price_to_sales_ratio": ps_ratio
    }

def analyze_margin_trends(metrics: list) -> dict:
    """Analyzes historical margin trends."""
    
    gross_margins = [m.gross_margin for m in metrics]
    operating_margins = [m.operating_margin for m in metrics]
    net_margins = [m.net_margin for m in metrics]

    gm_trend = _calculate_trend(gross_margins)
    om_trend = _calculate_trend(operating_margins)
    nm_trend = _calculate_trend(net_margins)
    
    score = 0
    
    # Gross Margin
    if gross_margins[0] is not None:
        if gross_margins[0] > 0.5: # Healthy margin
            score += 0.2
        if gm_trend > 0: # Expanding
            score += 0.2

    # Operating Margin
    if operating_margins[0] is not None:
        if operating_margins[0] > 0.15: # Healthy margin
            score += 0.2
        if om_trend > 0: # Expanding
            score += 0.2
            
    # Net Margin Trend
    if nm_trend > 0:
        score += 0.2
        
    score = min(score, 1.0)
    
    return {
        "score": score,
        "gross_margin": gross_margins[0],
        "gross_margin_trend": gm_trend,
        "operating_margin": operating_margins[0],
        "operating_margin_trend": om_trend,
        "net_margin": net_margins[0],
        "net_margin_trend": nm_trend
    }

def analyze_insider_conviction(trades: list) -> dict:
    """Analyzes insider trading activity."""
    
    buys = sum(t.transaction_value for t in trades if t.transaction_value and t.transaction_shares > 0)
    sells = sum(abs(t.transaction_value) for t in trades if t.transaction_value and t.transaction_shares < 0)
    
    if (buys + sells) == 0:
        net_flow_ratio = 0
    else:
        net_flow_ratio = (buys - sells) / (buys + sells)
    
    score = 0
    if net_flow_ratio > 0.5:
        score = 1.0
    elif net_flow_ratio > 0.1:
        score = 0.7
    elif net_flow_ratio > -0.1:
        score = 0.5 # Neutral
    else:
        score = 0.2
        
    return {
        "score": score,
        "net_flow_ratio": net_flow_ratio,
        "buys": buys,
        "sells": sells
    }

def check_financial_health(metrics) -> dict:
    """Checks the company's financial health."""

    debt_to_equity = metrics.debt_to_equity
    current_ratio = metrics.current_ratio

    score = 1.0

    # Debt to Equity
    if debt_to_equity is not None:
        if debt_to_equity > 1.5:
            score -= 0.5
        elif debt_to_equity > 0.8:
            score -= 0.2

    # Current Ratio
    if current_ratio is not None:
        if current_ratio < 1.0:
            score -= 0.5
        elif current_ratio < 1.5:
            score -= 0.2

    score = max(score, 0.0)

    return {
        "score": score,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio
    }


def _fmt_pct(val) -> str:
    """Format a value as percentage string, handling None."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _build_reasoning_text(reasoning: dict) -> str:
    """Convert the structured growth reasoning dict to a concise Chinese text summary."""

    parts = []

    # -- historical_growth --
    hg = reasoning.get("historical_growth", {})
    hg_score = hg.get("score", 0)
    rev_g = hg.get("revenue_growth")
    eps_g = hg.get("eps_growth")
    details = f"评分{hg_score:.1f}"
    if rev_g is not None:
        details += f"（收入增长{_fmt_pct(rev_g)}"
        if eps_g is not None:
            details += f"，EPS增长{_fmt_pct(eps_g)}"
        details += "）"
    parts.append(f"历史增长：{details}")

    # -- growth_valuation --
    gv = reasoning.get("growth_valuation", {})
    gv_score = gv.get("score", 0)
    peg = gv.get("peg_ratio")
    parts.append(f"增长估值：评分{gv_score:.1f}（PEG={peg if peg is not None else 'N/A'}）")

    # -- margin_expansion --
    me = reasoning.get("margin_expansion", {})
    me_score = me.get("score", 0)
    gm = me.get("gross_margin")
    details = f"评分{me_score:.1f}"
    if gm is not None:
        details += f"（毛利率{_fmt_pct(gm)}）"
    parts.append(f"利润率扩张：{details}")

    # -- insider_conviction --
    ic = reasoning.get("insider_conviction", {})
    ic_score = ic.get("score", 0)
    parts.append(f"内部人信心：评分{ic_score:.1f}")

    # -- financial_health --
    fh = reasoning.get("financial_health", {})
    fh_score = fh.get("score", 0)
    de = fh.get("debt_to_equity")
    cr = fh.get("current_ratio")
    details = f"评分{fh_score:.1f}"
    if de is not None or cr is not None:
        details += "（"
        if de is not None:
            details += f"D/E={de:.2f}"
        if de is not None and cr is not None:
            details += "，"
        if cr is not None:
            details += f"流动比率{cr:.2f}"
        details += "）"
    parts.append(f"财务健康：{details}")

    # -- final_analysis --
    fa = reasoning.get("final_analysis", {})
    sig = fa.get("signal", "")
    conf = fa.get("confidence", 0)
    ws = fa.get("weighted_score", 0)
    sig_cn = {"bullish": "看涨", "bearish": "看跌", "neutral": "中性"}.get(sig, sig)
    parts.append(f"综合信号：{sig_cn}（置信度{conf}%，加权评分{ws:.2f}）")

    return "；".join(parts)
