from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import statistics
from src.utils.api_key import get_api_key_from_state

class PhilFisherSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def phil_fisher_agent(state: AgentState, agent_id: str = "phil_fisher_agent"):
    """
    Analyzes stocks using Phil Fisher's investing principles:
      - Seek companies with long-term above-average growth potential
      - Emphasize quality of management and R&D
      - Look for strong margins, consistent growth, and manageable leverage
      - Combine fundamental 'scuttlebutt' style checks with basic sentiment and insider data
      - Willing to pay up for quality, but still mindful of valuation
      - Generally focuses on long-term compounding

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    fisher_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "收集财务报表项目")
        # Include relevant line items for Phil Fisher's approach:
        #   - Growth & Quality: revenue, net_income, earnings_per_share, R&D expense
        #   - Margins & Stability: operating_income, operating_margin, gross_margin
        #   - Management Efficiency & Leverage: total_debt, shareholders_equity, free_cash_flow
        #   - Valuation: net_income, free_cash_flow (for P/E, P/FCF), ebit, ebitda
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "earnings_per_share",
                "free_cash_flow",
                "research_and_development",
                "operating_income",
                "operating_margin",
                "gross_margin",
                "total_debt",
                "shareholders_equity",
                "cash_and_equivalents",
                "ebit",
                "ebitda",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "获取市值")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取内部人交易数据")
        insider_trades = get_insider_trades(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取公司新闻")
        company_news = get_company_news(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "分析增长与质量")
        growth_quality = analyze_fisher_growth_quality(financial_line_items)

        progress.update_status(agent_id, ticker, "分析利润率与稳定性")
        margins_stability = analyze_margins_stability(financial_line_items)

        progress.update_status(agent_id, ticker, "分析管理效率与杠杆")
        mgmt_efficiency = analyze_management_efficiency_leverage(financial_line_items)

        progress.update_status(agent_id, ticker, "分析估值（费雪风格）")
        fisher_valuation = analyze_fisher_valuation(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "分析内部人活动")
        insider_activity = analyze_insider_activity(insider_trades)

        progress.update_status(agent_id, ticker, "分析市场情绪")
        sentiment_analysis = analyze_sentiment(company_news)

        # Combine partial scores with weights typical for Fisher:
        #   30% Growth & Quality
        #   25% Margins & Stability
        #   20% Management Efficiency
        #   15% Valuation
        #   5% Insider Activity
        #   5% Sentiment
        total_score = (
            growth_quality["score"] * 0.30
            + margins_stability["score"] * 0.25
            + mgmt_efficiency["score"] * 0.20
            + fisher_valuation["score"] * 0.15
            + insider_activity["score"] * 0.05
            + sentiment_analysis["score"] * 0.05
        )

        max_possible_score = 10

        # Simple bullish/neutral/bearish signal
        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "growth_quality": growth_quality,
            "margins_stability": margins_stability,
            "management_efficiency": mgmt_efficiency,
            "valuation_analysis": fisher_valuation,
            "insider_activity": insider_activity,
            "sentiment_analysis": sentiment_analysis,
        }

        progress.update_status(agent_id, ticker, "生成费雪风格分析")
        fisher_output = generate_fisher_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        fisher_analysis[ticker] = {
            "signal": fisher_output.signal,
            "confidence": fisher_output.confidence,
            "reasoning": fisher_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=fisher_output.reasoning)

    # Wrap results in a single message
    message = HumanMessage(content=json.dumps(fisher_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(fisher_analysis, "菲利普·费雪智能体")

    state["data"]["analyst_signals"][agent_id] = fisher_analysis

    progress.update_status(agent_id, None, "完成")
    
    return {"messages": [message], "data": state["data"]}


def analyze_fisher_growth_quality(financial_line_items: list) -> dict:
    """
    Evaluate growth & quality:
      - Consistent Revenue Growth
      - Consistent EPS Growth
      - R&D as a % of Revenue (if relevant, indicative of future-oriented spending)
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "财务数据不足以进行增长/质量分析",
        }

    details = []
    raw_score = 0  # up to 9 raw points => scale to 0–10

    # 1. Revenue Growth (annualized CAGR)
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        # Calculate annualized growth rate (CAGR) for proper comparison
        latest_rev = revenues[0]
        oldest_rev = revenues[-1]
        num_years = len(revenues) - 1
        if oldest_rev > 0 and latest_rev > 0:
            # CAGR formula: (ending_value/beginning_value)^(1/years) - 1
            rev_growth = (latest_rev / oldest_rev) ** (1 / num_years) - 1
            if rev_growth > 0.20:  # 20% annualized
                raw_score += 3
                details.append(f"年化收入增长非常强劲：{rev_growth:.1%}")
            elif rev_growth > 0.10:  # 10% annualized
                raw_score += 2
                details.append(f"年化收入增长中等：{rev_growth:.1%}")
            elif rev_growth > 0.03:  # 3% annualized
                raw_score += 1
                details.append(f"年化收入增长轻微：{rev_growth:.1%}")
            else:
                details.append(f"年化收入增长微乎其微或为负：{rev_growth:.1%}")
        else:
            details.append("最早收入为零/负数；无法计算增长。")
    else:
        details.append("收入数据点不足以进行增长计算。")

    # 2. EPS Growth (annualized CAGR)
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        oldest_eps = eps_values[-1]
        num_years = len(eps_values) - 1
        if oldest_eps > 0 and latest_eps > 0:
            # CAGR formula for EPS
            eps_growth = (latest_eps / oldest_eps) ** (1 / num_years) - 1
            if eps_growth > 0.20:  # 20% annualized
                raw_score += 3
                details.append(f"年化每股收益增长非常强劲：{eps_growth:.1%}")
            elif eps_growth > 0.10:  # 10% annualized
                raw_score += 2
                details.append(f"年化每股收益增长中等：{eps_growth:.1%}")
            elif eps_growth > 0.03:  # 3% annualized
                raw_score += 1
                details.append(f"年化每股收益增长轻微：{eps_growth:.1%}")
            else:
                details.append(f"年化每股收益增长微乎其微或为负：{eps_growth:.1%}")
        else:
            details.append("最早每股收益接近零；跳过每股收益增长计算。")
    else:
        details.append("每股收益数据点不足以进行增长计算。")

    # 3. R&D as % of Revenue (if we have R&D data)
    rnd_values = [fi.research_and_development for fi in financial_line_items if fi.research_and_development is not None]
    if rnd_values and revenues and len(rnd_values) == len(revenues):
        # We'll just look at the most recent for a simple measure
        recent_rnd = rnd_values[0]
        recent_rev = revenues[0] if revenues[0] else 1e-9
        rnd_ratio = recent_rnd / recent_rev
        # Generally, Fisher admired companies that invest aggressively in R&D,
        # but it must be appropriate. We'll assume "3%-15%" is healthy, just as an example.
        if 0.03 <= rnd_ratio <= 0.15:
            raw_score += 3
            details.append(f"研发比率 {rnd_ratio:.1%} 表明对未来增长有重大投入")
        elif rnd_ratio > 0.15:
            raw_score += 2
            details.append(f"研发比率 {rnd_ratio:.1%} 非常高（管理得当可能是好事）")
        elif rnd_ratio > 0.0:
            raw_score += 1
            details.append(f"研发比率 {rnd_ratio:.1%} 偏低但仍为正")
        else:
            details.append("无有意义的研发支出比率")
    else:
        details.append("研发数据不足以评估")

    # scale raw_score (max 9) to 0–10
    final_score = min(10, (raw_score / 9) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_margins_stability(financial_line_items: list) -> dict:
    """
    Looks at margin consistency (gross/operating margin) and general stability over time.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "数据不足以分析利润率稳定性",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0-10

    # 1. Operating Margin Consistency
    op_margins = [fi.operating_margin for fi in financial_line_items if fi.operating_margin is not None]
    if len(op_margins) >= 2:
        # Check if margins are stable or improving (comparing oldest to newest)
        oldest_op_margin = op_margins[-1]
        newest_op_margin = op_margins[0]
        if newest_op_margin >= oldest_op_margin > 0:
            raw_score += 2
            details.append(f"营业利润率稳定或改善中（{oldest_op_margin:.1%} -> {newest_op_margin:.1%}）")
        elif newest_op_margin > 0:
            raw_score += 1
            details.append(f"营业利润率为正但略有下降")
        else:
            details.append(f"营业利润率可能为负或不确定")
    else:
        details.append("营业利润率数据点不足")

    # 2. Gross Margin Level
    gm_values = [fi.gross_margin for fi in financial_line_items if fi.gross_margin is not None]
    if gm_values:
        # We'll just take the most recent
        recent_gm = gm_values[0]
        if recent_gm > 0.5:
            raw_score += 2
            details.append(f"毛利率强劲：{recent_gm:.1%}")
        elif recent_gm > 0.3:
            raw_score += 1
            details.append(f"毛利率中等：{recent_gm:.1%}")
        else:
            details.append(f"毛利率偏低：{recent_gm:.1%}")
    else:
        details.append("无毛利率数据")

    # 3. Multi-year Margin Stability
    #   e.g. if we have at least 3 data points, see if standard deviation is low.
    if len(op_margins) >= 3:
        stdev = statistics.pstdev(op_margins)
        if stdev < 0.02:
            raw_score += 2
            details.append("营业利润率在多年间极为稳定")
        elif stdev < 0.05:
            raw_score += 1
            details.append("营业利润率较为稳定")
        else:
            details.append("营业利润率波动较大")
    else:
        details.append("利润率数据点不足以进行波动性检查")

    # scale raw_score (max 6) to 0-10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_efficiency_leverage(financial_line_items: list) -> dict:
    """
    Evaluate management efficiency & leverage:
      - Return on Equity (ROE)
      - Debt-to-Equity ratio
      - Possibly check if free cash flow is consistently positive
    """
    if not financial_line_items:
        return {
            "score": 0,
            "details": "无财务数据进行管理效率分析",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0–10

    # 1. Return on Equity (ROE)
    ni_values = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    eq_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]
    if ni_values and eq_values and len(ni_values) == len(eq_values):
        recent_ni = ni_values[0]
        recent_eq = eq_values[0] if eq_values[0] else 1e-9
        if recent_ni > 0:
            roe = recent_ni / recent_eq
            if roe > 0.2:
                raw_score += 3
                details.append(f"净资产收益率高：{roe:.1%}")
            elif roe > 0.1:
                raw_score += 2
                details.append(f"净资产收益率中等：{roe:.1%}")
            elif roe > 0:
                raw_score += 1
                details.append(f"净资产收益率为正但偏低：{roe:.1%}")
            else:
                details.append(f"净资产收益率接近零或为负：{roe:.1%}")
        else:
            details.append("最近净利润为零或为负，影响净资产收益率")
    else:
        details.append("数据不足以计算净资产收益率")

    # 2. Debt-to-Equity
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    if debt_values and eq_values and len(debt_values) == len(eq_values):
        recent_debt = debt_values[0]
        recent_equity = eq_values[0] if eq_values[0] else 1e-9
        dte = recent_debt / recent_equity
        if dte < 0.3:
            raw_score += 2
            details.append(f"债务股本比低：{dte:.2f}")
        elif dte < 1.0:
            raw_score += 1
            details.append(f"债务股本比可控：{dte:.2f}")
        else:
            details.append(f"债务股本比偏高：{dte:.2f}")
    else:
        details.append("数据不足以进行债务股本比分析")

    # 3. FCF Consistency
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    if fcf_values and len(fcf_values) >= 2:
        # Check if FCF is positive in recent years
        positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
        # We'll be simplistic: if most are positive, reward
        ratio = positive_fcf_count / len(fcf_values)
        if ratio > 0.8:
            raw_score += 1
            details.append(f"多数期间自由现金流为正（{positive_fcf_count}/{len(fcf_values)}）")
        else:
            details.append(f"自由现金流不稳定或经常为负")
    else:
        details.append("自由现金流数据不足或无数据以检查一致性")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_fisher_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Phil Fisher is willing to pay for quality and growth, but still checks:
      - P/E
      - P/FCF
      - (Optionally) Enterprise Value metrics, but simpler approach is typical
    We will grant up to 2 points for each of two metrics => max 4 raw => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "数据不足以进行估值"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 20:
            pe_points = 2
            details.append(f"市盈率合理有吸引力：{pe:.2f}")
        elif pe < 30:
            pe_points = 1
            details.append(f"市盈率偏高但可能有理可据：{pe:.2f}")
        else:
            details.append(f"市盈率非常高：{pe:.2f}")
        raw_score += pe_points
    else:
        details.append("无正净利润用于计算市盈率")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 20:
            pfcf_points = 2
            details.append(f"市现率合理：{pfcf:.2f}")
        elif pfcf < 30:
            pfcf_points = 1
            details.append(f"市现率偏高：{pfcf:.2f}")
        else:
            details.append(f"市现率过高：{pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("无正自由现金流用于计算市现率")

    # scale raw_score (max 4) to 0–10
    final_score = min(10, (raw_score / 4) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, we nudge the score up.
      - If there's mostly selling, we reduce it.
      - Otherwise, neutral.
    """
    # Default is neutral (5/10).
    score = 5
    details = []

    if not insider_trades:
        details.append("无内部人交易数据；默认中性")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        if trade.transaction_shares is not None:
            if trade.transaction_shares > 0:
                buys += 1
            elif trade.transaction_shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("未发现买卖交易；中性")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"内部人大量买入：{buys} 次买入对比 {sells} 次卖出")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"内部人适度买入：{buys} 次买入对比 {sells} 次卖出")
    else:
        score = 4
        details.append(f"内部人主要卖出：{buys} 次买入对比 {sells} 次卖出")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment: negative keyword check vs. overall volume.
    """
    if not news_items:
        return {"score": 5, "details": "无新闻数据；默认中性情绪"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title_lower = (news.title or "").lower()
        if any(word in title_lower for word in negative_keywords):
            negative_count += 1

    details = []
    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"负面头条比例高：{negative_count}/{len(news_items)}")
    elif negative_count > 0:
        score = 6
        details.append(f"部分负面头条：{negative_count}/{len(news_items)}")
    else:
        score = 8
        details.append("头条以正面/中性为主")

    return {"score": score, "details": "; ".join(details)}


def generate_fisher_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> PhilFisherSignal:
    """
    Generates a JSON signal in the style of Phil Fisher.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
              "system",
              """你是菲利普·费雪AI智能体，使用他的投资原则进行决策:

              1. 重视长期增长潜力和管理层质量。
              2. 关注投资研发以开发未来产品/服务的公司。
              3. 寻找强劲的盈利能力和一致的利润率。
              4. 愿意为卓越公司支付溢价但仍关注估值。
              5. 依赖深入调研 (闲聊法) 和全面的基本面检查。

              在提供推理时，要详尽具体:
              1. 详细讨论公司的增长前景，引用具体的指标和趋势
              2. 评估管理层质量及其资本配置决策
              3. 突出研发投资和可能推动未来增长的产品管线
              4. 评估利润率和盈利指标的一致性，提供精确数字
              5. 解释可能维持3-5年以上增长的竞争优势
              6. 使用菲利普·费雪的有条理、注重增长和长期导向的语调

              例如，如果看涨: "该公司展现出我们寻求的持续增长特征，五年间收入以每年18%的速度增长。管理层展现了卓越的前瞻性，将15%的收入分配给研发，已产生三个有前景的新产品线。22-24%的一致营业利润率表明定价权和运营效率将持续..."

              例如，如果看跌: "尽管处于增长行业，管理层未能将研发投资 (仅占收入的5%) 转化为有意义的新产品。利润率在10-15%之间波动，显示运营执行不一致。公司面临来自三个拥有更优分销网络的更大竞争对手的日益激烈竞争。鉴于对长期增长可持续性的这些担忧..."

              用中文输出推理。你必须输出一个JSON对象:
                - "signal": "bullish" or "bearish" or "neutral"
                - "confidence": 0到100之间的浮点数
                - "reasoning": 详细解释
              """,
            ),
            (
              "human",
              """根据以下分析，创建费雪风格的投资信号。用中文回答。

              {ticker} 的分析数据:
              {analysis_data}

              按以下JSON格式返回交易信号:
              {{
                "signal": "bullish/bearish/neutral",
                "confidence": float (0-100),
                "reasoning": "string"
              }}
              """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return PhilFisherSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="分析出错，默认中性"
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=PhilFisherSignal,
        state=state,
        agent_name=agent_id,
        default_factory=create_default_signal,
    )
