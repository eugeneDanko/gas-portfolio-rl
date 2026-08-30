from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


@dataclass(frozen=True)
class TradeRecord:
    step: int
    calendar_date: str
    instrument: str
    side: str
    notional: float
    price: float
    fee: float


class LiveTradingDisplay:
    """Notebook-friendly live monitor for baseline and RL strategy runs."""

    def __init__(
        self,
        *,
        refresh_every: int = 2,
        window_size: int = 80,
        pause_seconds: float = 0.0,
        max_tape_rows: int = 10,
        live: bool = True,
        minimum_trade_notional: float = 1.0,
    ) -> None:
        if refresh_every < 1 or window_size < 2 or max_tape_rows < 1 or pause_seconds < 0:
            raise ValueError("Display intervals and tape size must be positive")
        self.refresh_every = refresh_every
        self.window_size = window_size
        self.pause_seconds = pause_seconds
        self.max_tape_rows = max_tape_rows
        self.live = live
        self.minimum_trade_notional = minimum_trade_notional
        self.strategy_name = "strategy"
        self.trajectory: list[dict[str, Any]] = []
        self.trades: list[TradeRecord] = []
        self._live_figure: Figure | None = None
        self._live_axes: np.ndarray | None = None
        self._live_lines: dict[str, Any] = {}
        self._display_handle: Any = None
        self._live_initialized = False

    def reset(self, strategy_name: str, info: Mapping[str, Any]) -> None:
        self.strategy_name = strategy_name
        self.trajectory = [dict(info)]
        self.trades = []
        self._live_figure = None
        self._live_axes = None
        self._live_lines = {}
        self._display_handle = None
        self._live_initialized = False
        if self.live:
            self._ensure_live_figure()
            self._update_figure(
                self._live_figure,
                self._live_axes,
                self._live_lines,
                full_history=False,
                smooth_limits=False,
            )
            self._show_live()

    def update(
        self,
        strategy_name: str,
        info: Mapping[str, Any],
        action: np.ndarray,
        done: bool,
    ) -> None:
        del action
        self.strategy_name = strategy_name
        row = dict(info)
        self.trajectory.append(row)
        self._record_trades(row)
        if self.live and (done or int(row["step"]) % self.refresh_every == 0):
            self._ensure_live_figure()
            self._update_figure(
                self._live_figure,
                self._live_axes,
                self._live_lines,
                full_history=False,
                smooth_limits=self._live_initialized,
            )
            self._live_initialized = True
            self._show_live()

    def render(self, *, full_history: bool = True) -> Figure:
        if not self.trajectory:
            raise ValueError("Call reset() before render()")
        figure, axes, lines = self._create_figure()
        self._update_figure(
            figure,
            axes,
            lines,
            full_history=full_history,
            smooth_limits=False,
        )
        return figure

    def _ensure_live_figure(self) -> None:
        if self._live_figure is None:
            self._live_figure, self._live_axes, self._live_lines = self._create_figure()

    def _create_figure(self) -> tuple[Figure, np.ndarray, dict[str, Any]]:
        figure = Figure(figsize=(15, 11))
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 2)
        figure.subplots_adjust(top=0.91, bottom=0.07, hspace=0.36, wspace=0.18)

        lines: dict[str, Any] = {}
        lines["stock_price"], = axes[0, 0].plot([], [], label="GasCo")
        lines["futures_price"], = axes[0, 0].plot([], [], label="Front future")
        axes[0, 0].set_title("Normalized market prices")
        axes[0, 0].set_ylabel("Index, episode start = 1")
        axes[0, 0].legend(loc="best")

        lines["pnl"], = axes[0, 1].plot([], [], label="Cumulative P&L", color="tab:blue")
        axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[0, 1].set_title("Cumulative P&L")
        axes[0, 1].set_ylabel("Money")

        lines["cash"], = axes[1, 0].plot([], [], label="Free cash", color="tab:green")
        axes[1, 0].set_title("Remaining cash")
        axes[1, 0].set_ylabel("Money")

        lines["fees"], = axes[1, 1].plot(
            [], [], label="Cumulative commissions", color="tab:red"
        )
        lines["dividends"], = axes[1, 1].plot(
            [], [], label="Cumulative dividends", color="tab:purple"
        )
        axes[1, 1].set_title("Cash flows")
        axes[1, 1].set_ylabel("Money")
        axes[1, 1].legend(loc="best")

        lines["stock_weight"], = axes[2, 0].plot([], [], label="Stock weight")
        lines["futures_exposure"], = axes[2, 0].plot([], [], label="Futures exposure")
        lines["cash_weight"], = axes[2, 0].plot([], [], label="Cash weight")
        axes[2, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[2, 0].set_title("Portfolio allocation")
        axes[2, 0].set_ylabel("Weight / exposure")
        axes[2, 0].legend(loc="best")

        for axis in axes.flat[:-1]:
            axis.grid(alpha=0.25)
            axis.set_xlabel("Trading step")
        return figure, axes, lines

    def _update_figure(
        self,
        figure: Figure,
        axes: np.ndarray,
        lines: dict[str, Any],
        *,
        full_history: bool,
        smooth_limits: bool,
    ) -> None:
        all_rows = self.trajectory
        latest_step = int(all_rows[-1]["step"])
        if full_history:
            rows = all_rows
            x_left = int(all_rows[0]["step"])
            x_right = max(latest_step, x_left + 1)
        else:
            first_visible_step = max(0, latest_step - self.window_size + 1)
            rows = [row for row in all_rows if int(row["step"]) >= first_visible_step]
            x_left = first_visible_step
            x_right = x_left + self.window_size - 1

        step = np.asarray([row["step"] for row in rows])
        stock_price = np.asarray([row["stock_price"] for row in rows], dtype=float)
        futures_price = np.asarray([row["futures_price"] for row in rows], dtype=float)
        pnl = np.asarray([row["pnl"] for row in rows], dtype=float)
        cash = np.asarray([row["cash"] for row in rows], dtype=float)
        fees = np.asarray([row["cumulative_transaction_cost"] for row in rows], dtype=float)
        dividends = np.asarray([row.get("cumulative_dividends", 0.0) for row in rows])
        stock_weight = np.asarray([row["stock_weight"] for row in rows], dtype=float)
        futures_exposure = np.asarray(
            [row["futures_exposure"] for row in rows], dtype=float
        )
        cash_weight = np.asarray([row["cash_weight"] for row in rows], dtype=float)
        initial_stock_price = float(all_rows[0]["stock_price"])
        initial_futures_price = float(all_rows[0]["futures_price"])

        latest = all_rows[-1]
        volatility = latest.get("reward_components", {}).get("portfolio_volatility", 0.0)
        title = (
            f"{self.strategy_name} | step {latest['step']} | "
            f"value {latest['portfolio_value']:,.0f} | P&L {latest['pnl']:+,.0f} | "
            f"cash {latest['cash']:,.0f} | DD {latest['drawdown']:.1%} | "
            f"vol {volatility:.1%} | margin {latest['margin_ratio']:.2f} | "
            f"reward {latest.get('cumulative_reward', 0.0):+.3f}"
        )
        if figure._suptitle is None:
            figure.suptitle(title, fontsize=12)
        else:
            figure._suptitle.set_text(title)

        series = {
            "stock_price": stock_price / initial_stock_price,
            "futures_price": futures_price / initial_futures_price,
            "pnl": pnl,
            "cash": cash,
            "fees": fees,
            "dividends": dividends,
            "stock_weight": stock_weight,
            "futures_exposure": futures_exposure,
            "cash_weight": cash_weight,
        }
        for name, values in series.items():
            lines[name].set_data(step, values)

        for axis in axes.flat[:-1]:
            axis.set_xlim(x_left, x_right)
        _set_y_limits(
            axes[0, 0],
            np.concatenate([series["stock_price"], series["futures_price"]]),
            smooth_limits,
        )
        _set_y_limits(axes[0, 1], pnl, smooth_limits, include_zero=True)
        _set_y_limits(axes[1, 0], cash, smooth_limits)
        _set_y_limits(
            axes[1, 1],
            np.concatenate([fees, dividends]),
            smooth_limits,
            include_zero=True,
        )
        _set_y_limits(
            axes[2, 0],
            np.concatenate([stock_weight, futures_exposure, cash_weight]),
            smooth_limits,
            include_zero=True,
        )

        tape_axis = axes[2, 1]
        tape_axis.clear()
        tape_axis.axis("off")
        tape_axis.set_title("Latest strategy trades")
        recent = self.trades[-self.max_tape_rows :]
        if recent:
            cells = [
                [
                    trade.calendar_date,
                    trade.instrument,
                    trade.side,
                    f"{trade.notional:,.0f}",
                    f"{trade.price:,.2f}",
                    f"{trade.fee:,.2f}",
                ]
                for trade in reversed(recent)
            ]
            table = tape_axis.table(
                cellText=cells,
                colLabels=["Date", "Instrument", "Side", "Notional", "Price", "Fee"],
                loc="center",
                cellLoc="right",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.25)
        else:
            tape_axis.text(0.5, 0.5, "No trades yet", ha="center", va="center")
        figure.canvas.draw()

    def save(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure = self.render(full_history=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        return destination

    def _record_trades(self, info: Mapping[str, Any]) -> None:
        self._append_trade(
            info,
            "GasCo",
            float(info.get("stock_trade_value", 0.0)),
            float(info["stock_price"]),
            float(info.get("stock_transaction_cost", 0.0)),
        )
        self._append_trade(
            info,
            "Gas future",
            float(info.get("futures_trade_notional", 0.0)),
            float(info["futures_price"]),
            float(info.get("futures_transaction_cost", 0.0)),
        )
        forced = float(info.get("forced_liquidation_notional", 0.0))
        if forced >= self.minimum_trade_notional:
            self.trades.append(
                TradeRecord(
                    int(info["step"]),
                    str(info.get("calendar_date") or info["step"]),
                    "Gas future",
                    "LIQUIDATE",
                    forced,
                    float(info["futures_price"]),
                    float(info.get("forced_liquidation_cost", 0.0)),
                )
            )

    def _append_trade(
        self,
        info: Mapping[str, Any],
        instrument: str,
        signed_notional: float,
        price: float,
        fee: float,
    ) -> None:
        if abs(signed_notional) < self.minimum_trade_notional:
            return
        self.trades.append(
            TradeRecord(
                int(info["step"]),
                str(info.get("calendar_date") or info["step"]),
                instrument,
                "BUY" if signed_notional > 0 else "SELL",
                abs(signed_notional),
                price,
                fee,
            )
        )

    def _show_live(self) -> None:
        assert self._live_figure is not None
        try:
            from IPython.display import display

            if self._display_handle is None:
                self._display_handle = display(self._live_figure, display_id=True)
            elif hasattr(self._display_handle, "update"):
                self._display_handle.update(self._live_figure)
        except ImportError:
            self._live_figure.canvas.draw()
        if self.pause_seconds > 0:
            time.sleep(self.pause_seconds)


def _set_y_limits(
    axis: Any,
    values: np.ndarray,
    smooth: bool,
    *,
    include_zero: bool = False,
) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    low = float(np.min(finite))
    high = float(np.max(finite))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    padding = max(span * 0.08, max(abs(low), abs(high), 1.0) * 0.02)
    target_low = low - padding
    target_high = high + padding
    if not smooth:
        axis.set_ylim(target_low, target_high)
        return
    current_low, current_high = axis.get_ylim()
    contraction = 0.20
    next_low = min(target_low, current_low + contraction * (target_low - current_low))
    next_high = max(target_high, current_high + contraction * (target_high - current_high))
    axis.set_ylim(next_low, next_high)
