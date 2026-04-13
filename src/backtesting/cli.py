from __future__ import annotations

import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse

from colorama import Fore, Style, init
import questionary

from .engine import BacktestEngine
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.analysts import ANALYST_ORDER
from src.main import run_hedge_fund
from src.utils.ollama import ensure_ollama_and_model


def main() -> int:
    parser = argparse.ArgumentParser(description="运行回测引擎（模块化）")
    parser.add_argument("--tickers", type=str, required=False, help="逗号分隔的股票代码")
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="开始日期 YYYY-MM-DD",
    )
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--analysts", type=str, required=False)
    parser.add_argument("--analysts-all", action="store_true")
    parser.add_argument("--ollama", action="store_true")

    args = parser.parse_args()
    init(autoreset=True)

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else []

    # Analysts selection is simplified; no interactive prompts here
    if args.analysts_all:
        selected_analysts = [a[1] for a in ANALYST_ORDER]
    elif args.analysts:
        selected_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()]
    else:
        # Interactive analyst selection (same as legacy backtester)
        choices = questionary.checkbox(
            "使用空格键选择/取消选择分析师。",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\n按 'a' 全选/全不选。\n\n选好后按回车运行。",
            validate=lambda x: len(x) > 0 or "至少选择一位分析师。",
            style=questionary.Style(
                [
                    ("checkbox-selected", "fg:green"),
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "noinherit"),
                    ("pointer", "noinherit"),
                ]
            ),
        ).ask()
        if not choices:
            print("\n\n已中断。正在退出...")
            return 1
        selected_analysts = choices
        print(
            f"\n已选分析师: "
            f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n"
        )

    # Model selection simplified: default to first ordered model or Ollama flag
    if args.ollama:
        print(f"{Fore.CYAN}使用 Ollama 进行本地 LLM 推理。{Style.RESET_ALL}")
        model_name = questionary.select(
            "选择您的 Ollama 模型:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()
        if not model_name:
            print("\n\n已中断。正在退出...")
            return 1
        if model_name == "-":
            model_name = questionary.text("输入自定义模型名称:").ask()
            if not model_name:
                print("\n\n已中断。正在退出...")
                return 1
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}无法在缺少 Ollama 和所选模型的情况下继续。{Style.RESET_ALL}")
            return 1
        model_provider = ModelProvider.OLLAMA.value
        print(
            f"\n已选 {Fore.CYAN}Ollama{Style.RESET_ALL} 模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n"
        )
    else:
        model_choice = questionary.select(
            "选择您的 LLM 模型:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()
        if not model_choice:
            print("\n\n已中断。正在退出...")
            return 1
        model_name, model_provider = model_choice
        model_info = get_model_info(model_name, model_provider)
        if model_info and model_info.is_custom():
            model_name = questionary.text("输入自定义模型名称:").ask()
            if not model_name:
                print("\n\n已中断。正在退出...")
                return 1
        print(
            f"\n已选 {Fore.CYAN}{model_provider}{Style.RESET_ALL} 模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n"
        )

    engine = BacktestEngine(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
    )

    metrics = engine.run_backtest()
    values = engine.get_portfolio_values()

    # Minimal terminal output (no plots)
    if values:
        print(f"\n{Fore.WHITE}{Style.BRIGHT}引擎运行完成{Style.RESET_ALL}")
        last_value = values[-1]["Portfolio Value"]
        start_value = values[0]["Portfolio Value"]
        total_return = (last_value / start_value - 1.0) * 100.0 if start_value else 0.0
        print(f"总收益率: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
    if metrics.get("sharpe_ratio") is not None:
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    if metrics.get("sortino_ratio") is not None:
        print(f"索提诺比率: {metrics['sortino_ratio']:.2f}")
    if metrics.get("max_drawdown") is not None:
        md = abs(metrics["max_drawdown"]) if metrics["max_drawdown"] is not None else 0.0
        if metrics.get("max_drawdown_date"):
            print(f"最大回撤: {md:.2f}%（{metrics['max_drawdown_date']}）")
        else:
            print(f"最大回撤: {md:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




