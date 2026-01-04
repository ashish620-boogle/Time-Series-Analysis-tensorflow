from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import List, Literal


Signal = Literal["buy", "sell", "hold"]


@dataclass
class Trade:
    timestamp: dt.datetime
    action: Signal
    price: float
    size: float
    cash_after: float
    holdings_after: float
    realized_pnl: float


@dataclass
class Portfolio:
    cash: float = 10_000.0
    position: float = 0.0  # number of shares
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    history: List[Trade] = field(default_factory=list)

    def value(self, current_price: float) -> float:
        return self.cash + self.position * current_price

    def to_dict(self, current_price: float) -> dict:
        return {
            "cash": self.cash,
            "position_shares": self.position,
            "avg_price": self.avg_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(current_price),
            "total_value": self.value(current_price),
        }

    def unrealized_pnl(self, current_price: float) -> float:
        if self.position == 0:
            return 0.0
        return (current_price - self.avg_price) * self.position


def trading_signal(predicted_price: float, current_price: float, threshold: float = 0.002) -> Signal:
    """Return a basic long/flat signal using a relative improvement threshold."""
    change = (predicted_price - current_price) / current_price
    if change > threshold:
        return "buy"
    if change < -threshold:
        return "sell"
    return "hold"


def execute_signal(portfolio: Portfolio, signal: Signal, price: float, fraction: float = 0.001) -> Portfolio:
    now = dt.datetime.utcnow()
    if signal == "buy" and portfolio.cash > 0:
        budget = portfolio.cash * fraction
        size = budget / price
        new_cost = portfolio.avg_price * portfolio.position + size * price
        new_position = portfolio.position + size
        portfolio.avg_price = new_cost / new_position
        portfolio.position = new_position
        portfolio.cash -= budget
        portfolio.history.append(
            Trade(now, "buy", price, size, portfolio.cash, portfolio.position * price, portfolio.realized_pnl)
        )
    elif signal == "sell" and portfolio.position > 0:
        proceeds = portfolio.position * price
        portfolio.realized_pnl += (price - portfolio.avg_price) * portfolio.position
        portfolio.cash += proceeds
        portfolio.history.append(
            Trade(now, "sell", price, -portfolio.position, portfolio.cash, 0.0, portfolio.realized_pnl)
        )
        portfolio.position = 0.0
        portfolio.avg_price = 0.0
    return portfolio


def format_history(portfolio: Portfolio) -> List[dict]:
    return [asdict(trade) for trade in portfolio.history]
