from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
    get_prices,
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

class StanleyDruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def stanley_druckenmiller_agent(state: AgentState, agent_id: str = "stanley_druckenmiller_agent"):
    """
    Analyzes stocks using Stanley Druckenmiller's investing principles:
      - Seeking asymmetric risk-reward opportunities
      - Emphasizing growth, momentum, and sentiment
      - Willing to be aggressive if conditions are favorable
      - Focus on preserving capital by avoiding high-risk, low-reward bets

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    druck_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取财务科目")
        # Include relevant line items for Stan Druckenmiller's approach:
        #   - Growth & momentum: revenue, EPS, operating_income, ...
        #   - Valuation: net_income, free_cash_flow, ebit, ebitda
        #   - Leverage: total_debt, shareholders_equity
        #   - Liquidity: cash_and_equivalents
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "earnings_per_share",
                "net_income",
                "operating_income",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
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

        progress.update_status(agent_id, ticker, "获取内部人交易")
        insider_trades = get_insider_trades(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取公司新闻")
        company_news = get_company_news(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "获取近期价格数据以分析动量")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "分析增长与动量")
        growth_momentum_analysis = analyze_growth_and_momentum(financial_line_items, prices)

        progress.update_status(agent_id, ticker, "分析市场情绪")
        sentiment_analysis = analyze_sentiment(company_news)

        progress.update_status(agent_id, ticker, "分析内部人活动")
        insider_activity = analyze_insider_activity(insider_trades)

        progress.update_status(agent_id, ticker, "分析风险收益")
        risk_reward_analysis = analyze_risk_reward(financial_line_items, prices)

        progress.update_status(agent_id, ticker, "进行德鲁肯米勒式估值")
        valuation_analysis = analyze_druckenmiller_valuation(financial_line_items, market_cap)

        # Combine partial scores with weights typical for Druckenmiller:
        #   35% Growth/Momentum, 20% Risk/Reward, 20% Valuation,
        #   15% Sentiment, 10% Insider Activity = 100%
        total_score = (
            growth_momentum_analysis["score"] * 0.35
            + risk_reward_analysis["score"] * 0.20
            + valuation_analysis["score"] * 0.20
            + sentiment_analysis["score"] * 0.15
            + insider_activity["score"] * 0.10
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
            "growth_momentum_analysis": growth_momentum_analysis,
            "sentiment_analysis": sentiment_analysis,
            "insider_activity": insider_activity,
            "risk_reward_analysis": risk_reward_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status(agent_id, ticker, "生成德鲁肯米勒分析")
        druck_output = generate_druckenmiller_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        druck_analysis[ticker] = {
            "signal": druck_output.signal,
            "confidence": druck_output.confidence,
            "reasoning": druck_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=druck_output.reasoning)

    # 将结果封装为单条消息
    message = HumanMessage(content=json.dumps(druck_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(druck_analysis, "斯坦利·德鲁肯米勒智能体")

    state["data"]["analyst_signals"][agent_id] = druck_analysis

    progress.update_status(agent_id, None, "完成")
    
    return {"messages": [message], "data": state["data"]}


def analyze_growth_and_momentum(financial_line_items: list, prices: list) -> dict:
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "增长分析财务数据不足"}

    details = []
    raw_score = 0  # We'll sum up a maximum of 9 raw points, then scale to 0–10

    #
    # 1. Revenue Growth (annualized CAGR)
    #
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        latest_rev = revenues[0]
        older_rev = revenues[-1]
        num_years = len(revenues) - 1
        if older_rev > 0 and latest_rev > 0:
            # CAGR formula: (ending_value/beginning_value)^(1/years) - 1
            rev_growth = (latest_rev / older_rev) ** (1 / num_years) - 1
            if rev_growth > 0.08:  # 年化8%
                raw_score += 3
                details.append(f"强劲的年化收入增长：{rev_growth:.1%}")
            elif rev_growth > 0.04:  # 年化4%
                raw_score += 2
                details.append(f"中等的年化收入增长：{rev_growth:.1%}")
            elif rev_growth > 0.01:  # 年化1%
                raw_score += 1
                details.append(f"轻微的年化收入增长：{rev_growth:.1%}")
            else:
                details.append(f"收入增长极低或为负：{rev_growth:.1%}")
        else:
            details.append("早期收入为零或为负；无法计算收入增长。")
    else:
        details.append("收入数据点不足以计算增长。")

    #
    # 2. EPS Growth (annualized CAGR)
    #
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        num_years = len(eps_values) - 1
        # Calculate CAGR for positive EPS values
        if older_eps > 0 and latest_eps > 0:
            # CAGR formula for EPS
            eps_growth = (latest_eps / older_eps) ** (1 / num_years) - 1
            if eps_growth > 0.08:  # 年化8%
                raw_score += 3
                details.append(f"强劲的年化每股收益增长：{eps_growth:.1%}")
            elif eps_growth > 0.04:  # 年化4%
                raw_score += 2
                details.append(f"中等的年化每股收益增长：{eps_growth:.1%}")
            elif eps_growth > 0.01:  # 年化1%
                raw_score += 1
                details.append(f"轻微的年化每股收益增长：{eps_growth:.1%}")
            else:
                details.append(f"每股收益增长极低或为负：{eps_growth:.1%}")
        else:
            details.append("早期每股收益接近零；跳过每股收益增长计算。")
    else:
        details.append("每股收益数据点不足以计算增长。")

    #
    # 3. Price Momentum
    #
    # We'll give up to 3 points for strong momentum
    if prices and len(prices) > 30:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        if len(close_prices) >= 2:
            start_price = close_prices[0]
            end_price = close_prices[-1]
            if start_price > 0:
                pct_change = (end_price - start_price) / start_price
                if pct_change > 0.50:
                    raw_score += 3
                    details.append(f"非常强劲的价格动量：{pct_change:.1%}")
                elif pct_change > 0.20:
                    raw_score += 2
                    details.append(f"中等的价格动量：{pct_change:.1%}")
                elif pct_change > 0:
                    raw_score += 1
                    details.append(f"轻微的正向动量：{pct_change:.1%}")
                else:
                    details.append(f"负向价格动量：{pct_change:.1%}")
            else:
                details.append("起始价格无效（<=0）；无法计算动量。")
        else:
            details.append("价格数据不足以计算动量。")
    else:
        details.append("近期价格数据不足以进行动量分析。")

    # We assigned up to 3 points each for:
    #   revenue growth, eps growth, momentum
    # => max raw_score = 9
    # Scale to 0–10
    final_score = min(10, (raw_score / 9) * 10)

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
        details.append("无内部人交易数据；默认为中性")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        # Use transaction_shares to determine if it's a buy or sell
        # Negative shares = sell, positive shares = buy
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
        # 大量买入 => 从中性的5加3分 => 8
        score = 8
        details.append(f"大量内部人买入：{buys} 次买入 vs. {sells} 次卖出")
    elif buy_ratio > 0.4:
        # 中等买入 => +1 => 6
        score = 6
        details.append(f"中等内部人买入：{buys} 次买入 vs. {sells} 次卖出")
    else:
        # 较少内部人买入 => -1 => 4
        score = 4
        details.append(f"以内部人卖出为主：{buys} 次买入 vs. {sells} 次卖出")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment: negative keyword check vs. overall volume.
    """
    if not news_items:
        return {"score": 5, "details": "无新闻数据；默认为中性情绪"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title_lower = (news.title or "").lower()
        if any(word in title_lower for word in negative_keywords):
            negative_count += 1

    details = []
    if negative_count > len(news_items) * 0.3:
        # 超过30%为负面 => 偏看跌 => 3/10
        score = 3
        details.append(f"负面头条比例较高：{negative_count}/{len(news_items)}")
    elif negative_count > 0:
        # 有一些负面 => 6/10
        score = 6
        details.append(f"存在部分负面头条：{negative_count}/{len(news_items)}")
    else:
        # 以正面为主 => 8/10
        score = 8
        details.append("以正面/中性头条为主")

    return {"score": score, "details": "; ".join(details)}


def analyze_risk_reward(financial_line_items: list, prices: list) -> dict:
    """
    Assesses risk via:
      - Debt-to-Equity
      - Price Volatility
    Aims for strong upside with contained downside.
    """
    if not financial_line_items or not prices:
        return {"score": 0, "details": "风险收益分析数据不足"}

    details = []
    raw_score = 0  # We'll accumulate up to 6 raw points, then scale to 0-10

    #
    # 1. Debt-to-Equity
    #
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    equity_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]

    if debt_values and equity_values and len(debt_values) == len(equity_values) and len(debt_values) > 0:
        recent_debt = debt_values[0]
        recent_equity = equity_values[0] if equity_values[0] else 1e-9
        de_ratio = recent_debt / recent_equity
        if de_ratio < 0.3:
            raw_score += 3
            details.append(f"低负债权益比：{de_ratio:.2f}")
        elif de_ratio < 0.7:
            raw_score += 2
            details.append(f"中等负债权益比：{de_ratio:.2f}")
        elif de_ratio < 1.5:
            raw_score += 1
            details.append(f"偏高负债权益比：{de_ratio:.2f}")
        else:
            details.append(f"高负债权益比：{de_ratio:.2f}")
    else:
        details.append("无可用的负债/权益数据。")

    #
    # 2. Price Volatility
    #
    if len(prices) > 10:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        if len(close_prices) > 10:
            daily_returns = []
            for i in range(1, len(close_prices)):
                prev_close = close_prices[i - 1]
                if prev_close > 0:
                    daily_returns.append((close_prices[i] - prev_close) / prev_close)
            if daily_returns:
                stdev = statistics.pstdev(daily_returns)  # population stdev
                if stdev < 0.01:
                    raw_score += 3
                    details.append(f"低波动率：日收益率标准差 {stdev:.2%}")
                elif stdev < 0.02:
                    raw_score += 2
                    details.append(f"中等波动率：日收益率标准差 {stdev:.2%}")
                elif stdev < 0.04:
                    raw_score += 1
                    details.append(f"高波动率：日收益率标准差 {stdev:.2%}")
                else:
                    details.append(f"极高波动率：日收益率标准差 {stdev:.2%}")
            else:
                details.append("日收益率数据不足以计算波动率。")
        else:
            details.append("收盘价数据点不足以进行波动率分析。")
    else:
        details.append("价格数据不足以进行波动率分析。")

    # raw_score out of 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_druckenmiller_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "估值数据不足"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    ebit_values = [fi.ebit for fi in financial_line_items if fi.ebit is not None]
    ebitda_values = [fi.ebitda for fi in financial_line_items if fi.ebitda is not None]

    # For EV calculation, let's get the most recent total_debt & cash
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    cash_values = [fi.cash_and_equivalents for fi in financial_line_items if fi.cash_and_equivalents is not None]
    recent_debt = debt_values[0] if debt_values else 0
    recent_cash = cash_values[0] if cash_values else 0

    enterprise_value = market_cap + recent_debt - recent_cash

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 15:
            pe_points = 2
            details.append(f"有吸引力的市盈率：{pe:.2f}")
        elif pe < 25:
            pe_points = 1
            details.append(f"合理的市盈率：{pe:.2f}")
        else:
            details.append(f"偏高或极高的市盈率：{pe:.2f}")
        raw_score += pe_points
    else:
        details.append("无正净利润用于计算市盈率")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 15:
            pfcf_points = 2
            details.append(f"有吸引力的市现率：{pfcf:.2f}")
        elif pfcf < 25:
            pfcf_points = 1
            details.append(f"合理的市现率：{pfcf:.2f}")
        else:
            details.append(f"偏高或极高的市现率：{pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("无正自由现金流用于计算市现率")

    # 3) EV/EBIT
    recent_ebit = ebit_values[0] if ebit_values else None
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = enterprise_value / recent_ebit
        ev_ebit_points = 0
        if ev_ebit < 15:
            ev_ebit_points = 2
            details.append(f"有吸引力的EV/EBIT：{ev_ebit:.2f}")
        elif ev_ebit < 25:
            ev_ebit_points = 1
            details.append(f"合理的EV/EBIT：{ev_ebit:.2f}")
        else:
            details.append(f"偏高的EV/EBIT：{ev_ebit:.2f}")
        raw_score += ev_ebit_points
    else:
        details.append("EV <= 0 或 EBIT <= 0，无法计算EV/EBIT")

    # 4) EV/EBITDA
    recent_ebitda = ebitda_values[0] if ebitda_values else None
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = enterprise_value / recent_ebitda
        ev_ebitda_points = 0
        if ev_ebitda < 10:
            ev_ebitda_points = 2
            details.append(f"有吸引力的EV/EBITDA：{ev_ebitda:.2f}")
        elif ev_ebitda < 18:
            ev_ebitda_points = 1
            details.append(f"合理的EV/EBITDA：{ev_ebitda:.2f}")
        else:
            details.append(f"偏高的EV/EBITDA：{ev_ebitda:.2f}")
        raw_score += ev_ebitda_points
    else:
        details.append("EV <= 0 或 EBITDA <= 0，无法计算EV/EBITDA")

    # We have up to 2 points for each of the 4 metrics => 8 raw points max
    # Scale raw_score to 0–10
    final_score = min(10, (raw_score / 8) * 10)

    return {"score": final_score, "details": "; ".join(details)}


def generate_druckenmiller_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> StanleyDruckenmillerSignal:
    """
    Generates a JSON signal in the style of Stanley Druckenmiller.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
              "system",
              """你是斯坦利·德鲁肯米勒AI代理，使用他的投资原则做出决策：

              1. 寻求非对称的风险收益机会（大幅上行，有限下行）。
              2. 强调增长、动量和市场情绪。
              3. 通过避免重大回撤来保全资本。
              4. 愿意为真正的成长领军企业支付更高估值。
              5. 在确信度高的条件下可以激进。
              6. 如果投资逻辑改变，迅速止损。

              规则：
              - 奖励显示强劲收入/盈利增长和正向股票动量的公司。
              - 评估情绪和内部人活动作为支持或矛盾信号。
              - 注意高杠杆或威胁资本的极端波动率。
              - 输出包含signal、confidence和reasoning字符串的JSON对象。

              在提供推理时，请做到详尽且具体：
              1. 解释最影响你决策的增长和动量指标
              2. 用具体的数字证据突出风险收益特征
              3. 讨论可能推动价格走势的市场情绪和催化剂
              4. 同时阐述上行潜力和下行风险
              5. 提供相对于增长前景的具体估值背景
              6. 使用德鲁肯米勒果断、注重动量和信念驱动的语调

              例如，看涨时："该公司展现出卓越的动量，收入同比增速从22%加速至35%，股价在过去三个月上涨28%。风险收益高度非对称，基于自由现金流估值倍数扩张有70%的上行潜力，而强劲的资产负债表（现金为负债的3倍）仅面临15%的下行风险。内部人买入和积极的市场情绪提供了额外的推动力..."
              例如，看跌时："尽管近期股价有动量，但收入增长已从30%减速至12%，营业利润率正在收缩。风险收益比不利，上行潜力仅10%而下行风险达40%。竞争格局正在加剧，内部人卖出暗示信心减弱。我在其他地方看到了更好的机会..."

              用中文输出推理。
              """,
            ),
            (
              "human",
              """根据以下分析，创建一个德鲁肯米勒风格的投资信号。

              {ticker} 的分析数据：
              {analysis_data}

              请以以下JSON格式返回交易信号：
              {{
                "signal": "bullish/bearish/neutral",
                "confidence": 0到100之间的浮点数,
                "reasoning": "用中文写的推理字符串"
              }}
              """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return StanleyDruckenmillerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="分析出错，默认为中性"
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=StanleyDruckenmillerSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
