from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取内部人交易数据")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "分析交易模式")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "获取公司新闻")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status(agent_id, ticker, "综合信号")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)
        
        # Create structured reasoning similar to technical analysis
        reasoning = {
            "insider_trading": {
                "signal": "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else 
                         "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral",
                "confidence": round((max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100),
                "metrics": {
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                }
            },
            "news_sentiment": {
                "signal": "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else 
                         "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral",
                "confidence": round((max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100),
                "metrics": {
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                }
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": f"{'看涨' if bullish_signals > bearish_signals else '看跌' if bearish_signals > bullish_signals else '中性'}（基于加权信号比较）"
            }
        }

        reasoning_text = _build_reasoning_text(reasoning)
        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning_text,
            "reasoning_data": reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=reasoning_text)

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "市场情绪分析师")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "完成")

    return {
        "messages": [message],
        "data": data,
    }


def _signal_cn(sig: str) -> str:
    """Translate signal to Chinese."""
    return {"bullish": "看涨", "bearish": "看跌", "neutral": "中性"}.get(sig, sig)


def _build_reasoning_text(reasoning: dict) -> str:
    """Convert the structured sentiment reasoning dict to a concise Chinese text summary."""

    parts = []

    # -- Insider trading --
    insider = reasoning.get("insider_trading", {})
    insider_sig = _signal_cn(insider.get("signal", "neutral"))
    insider_metrics = insider.get("metrics", {})
    total_trades = insider_metrics.get("total_trades", 0)
    bullish_trades = insider_metrics.get("bullish_trades", 0)
    bearish_trades = insider_metrics.get("bearish_trades", 0)
    parts.append(f"内部人交易：{insider_sig}（{total_trades}笔交易，看涨{bullish_trades}笔，看跌{bearish_trades}笔）")

    # -- News sentiment --
    news = reasoning.get("news_sentiment", {})
    news_sig = _signal_cn(news.get("signal", "neutral"))
    news_metrics = news.get("metrics", {})
    total_articles = news_metrics.get("total_articles", 0)
    bullish_articles = news_metrics.get("bullish_articles", 0)
    bearish_articles = news_metrics.get("bearish_articles", 0)
    neutral_articles = news_metrics.get("neutral_articles", 0)
    parts.append(f"新闻情绪：{news_sig}（共{total_articles}篇，看涨{bullish_articles}篇，看跌{bearish_articles}篇，中性{neutral_articles}篇）")

    # -- Combined analysis --
    combined = reasoning.get("combined_analysis", {})
    total_bull = combined.get("total_weighted_bullish", 0)
    total_bear = combined.get("total_weighted_bearish", 0)
    determination = combined.get("signal_determination", "")
    parts.append(f"综合信号：{determination}（加权看涨{total_bull:.1f}，加权看跌{total_bear:.1f}）")

    return "；".join(parts)
