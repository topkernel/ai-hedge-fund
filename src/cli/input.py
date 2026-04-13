import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import questionary
from colorama import Fore, Style

from src.utils.analysts import ANALYST_ORDER
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider, find_model_by_name
from src.utils.ollama import ensure_ollama_and_model

from dataclasses import dataclass
from typing import Optional


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    require_tickers: bool = False,
    include_analyst_flags: bool = True,
    include_ollama: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tickers",
        type=str,
        required=require_tickers,
        help="逗号分隔的股票代码列表（例如：600418,000001）",
    )
    if include_analyst_flags:
        parser.add_argument(
            "--analysts",
            type=str,
            required=False,
            help="逗号分隔的分析师列表（例如：michael_burry,warren_buffett）",
        )
        parser.add_argument(
            "--analysts-all",
            action="store_true",
            help="使用所有可用分析师（覆盖 --analysts 参数）",
        )
    if include_ollama:
        parser.add_argument("--ollama", action="store_true", help="使用 Ollama 进行本地 LLM 推理")
    parser.add_argument("--model", type=str, required=False, help="指定使用的模型名称（例如：gpt-4o）")
    return parser


def add_date_args(parser: argparse.ArgumentParser, *, default_months_back: int | None = None) -> argparse.ArgumentParser:
    if default_months_back is None:
        parser.add_argument("--start-date", type=str, help="开始日期 (YYYY-MM-DD)")
        parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    else:
        parser.add_argument(
            "--end-date",
            type=str,
            default=datetime.now().strftime("%Y-%m-%d"),
            help="结束日期，格式为 YYYY-MM-DD",
        )
        parser.add_argument(
            "--start-date",
            type=str,
            default=(datetime.now() - relativedelta(months=default_months_back)).strftime("%Y-%m-%d"),
            help="开始日期，格式为 YYYY-MM-DD",
        )
    return parser


def parse_tickers(tickers_arg: str | None) -> list[str]:
    if not tickers_arg:
        return []
    return [ticker.strip() for ticker in tickers_arg.split(",") if ticker.strip()]


def select_analysts(flags: dict | None = None) -> list[str]:
    if flags and flags.get("analysts_all"):
        return [a[1] for a in ANALYST_ORDER]

    if flags and flags.get("analysts"):
        return [a.strip() for a in flags["analysts"].split(",") if a.strip()]

    choices = questionary.checkbox(
        "选择你的 AI 分析师",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\n说明：\n1. 按空格键选择/取消选择分析师。\n2. 按 'a' 键全选/取消全选。\n3. 完成后按回车键确认。",
        validate=lambda x: len(x) > 0 or "请至少选择一位分析师。",
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
        print("\n\n收到中断信号。正在退出...")
        sys.exit(0)

    print(
        f"\n已选择分析师: {', '.join(Fore.GREEN + c.title().replace('_', ' ') + Style.RESET_ALL for c in choices)}\n"
    )
    return choices


def select_model(use_ollama: bool, model_flag: str | None = None) -> tuple[str, str]:
    model_name: str = ""
    model_provider: str | None = None

    if model_flag:
        model = find_model_by_name(model_flag)
        if model:
            print(
                f"\n使用指定模型: {Fore.CYAN}{model.provider.value}{Style.RESET_ALL} - {Fore.GREEN + Style.BRIGHT}{model.model_name}{Style.RESET_ALL}\n"
            )
            return model.model_name, model.provider.value
        else:
            print(f"{Fore.RED}未找到模型 '{model_flag}'，请选择一个模型。{Style.RESET_ALL}")

    if use_ollama:
        print(f"{Fore.CYAN}使用 Ollama 进行本地 LLM 推理。{Style.RESET_ALL}")
        model_name = questionary.select(
            "选择你的 Ollama 模型：",
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
            print("\n\n收到中断信号。正在退出...")
            sys.exit(0)

        if model_name == "-":
            model_name = questionary.text("请输入自定义模型名称：").ask()
            if not model_name:
                print("\n\n收到中断信号。正在退出...")
                sys.exit(0)

        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}无法在缺少 Ollama 和所选模型的情况下继续运行。{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        print(
            f"\n已选择 {Fore.CYAN}Ollama{Style.RESET_ALL} 模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n"
        )
    else:
        model_choice = questionary.select(
            "选择你的 LLM 模型：",
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
            print("\n\n收到中断信号。正在退出...")
            sys.exit(0)

        model_name, model_provider = model_choice

        model_info = get_model_info(model_name, model_provider)
        if model_info and model_info.is_custom():
            model_name = questionary.text("请输入自定义模型名称：").ask()
            if not model_name:
                print("\n\n收到中断信号。正在退出...")
                sys.exit(0)

        if model_info:
            print(
                f"\n已选择 {Fore.CYAN}{model_provider}{Style.RESET_ALL} 模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n"
            )
        else:
            model_provider = "Unknown"
            print(f"\n已选择模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    return model_name, model_provider or ""


def resolve_dates(start_date: str | None, end_date: str | None, *, default_months_back: int | None = None) -> tuple[str, str]:
    if start_date:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("开始日期格式必须为 YYYY-MM-DD")
    if end_date:
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("结束日期格式必须为 YYYY-MM-DD")

    final_end = end_date or datetime.now().strftime("%Y-%m-%d")
    if start_date:
        final_start = start_date
    else:
        months = default_months_back if default_months_back is not None else 3
        end_date_obj = datetime.strptime(final_end, "%Y-%m-%d")
        final_start = (end_date_obj - relativedelta(months=months)).strftime("%Y-%m-%d")
    return final_start, final_end


@dataclass
class CLIInputs:
    tickers: list[str]
    selected_analysts: list[str]
    model_name: str
    model_provider: str
    start_date: str
    end_date: str
    initial_cash: float
    margin_requirement: float
    show_reasoning: bool = False
    show_agent_graph: bool = False
    raw_args: Optional[argparse.Namespace] = None


def parse_cli_inputs(
    *,
    description: str,
    require_tickers: bool,
    default_months_back: int | None,
    include_graph_flag: bool = False,
    include_reasoning_flag: bool = False,
) -> CLIInputs:
    parser = argparse.ArgumentParser(description=description)

    # Common/interactive flags
    add_common_args(parser, require_tickers=require_tickers, include_analyst_flags=True, include_ollama=True)
    add_date_args(parser, default_months_back=default_months_back)

    # Funding flags (standardized, with alias)
    parser.add_argument(
        "--initial-cash",
        "--initial-capital",
        dest="initial_cash",
        type=float,
        default=100000.0,
        help="初始现金头寸（别名：--initial-capital），默认 100000.0",
    )
    parser.add_argument(
        "--margin-requirement",
        dest="margin_requirement",
        type=float,
        default=0.0,
        help="做空初始保证金比例（例如 0.5 表示 50%%），默认 0.0",
    )

    if include_reasoning_flag:
        parser.add_argument("--show-reasoning", action="store_true", help="显示各分析师的推理过程")
    if include_graph_flag:
        parser.add_argument("--show-agent-graph", action="store_true", help="显示分析师关系图")

    args = parser.parse_args()

    # Normalize parsed values
    tickers = parse_tickers(getattr(args, "tickers", None))
    selected_analysts = select_analysts({
        "analysts_all": getattr(args, "analysts_all", False),
        "analysts": getattr(args, "analysts", None),
    })
    model_name, model_provider = select_model(getattr(args, "ollama", False), getattr(args, "model", None))
    start_date, end_date = resolve_dates(getattr(args, "start_date", None), getattr(args, "end_date", None), default_months_back=default_months_back)

    return CLIInputs(
        tickers=tickers,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
        start_date=start_date,
        end_date=end_date,
        initial_cash=getattr(args, "initial_cash", 100000.0),
        margin_requirement=getattr(args, "margin_requirement", 0.0),
        show_reasoning=getattr(args, "show_reasoning", False),
        show_agent_graph=getattr(args, "show_agent_graph", False),
        raw_args=args,
    )


