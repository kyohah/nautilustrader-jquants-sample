"""
NautilusTrader × J-Quants バックテストサンプル

BNF(小手川隆)スタイルの押し目買い戦略:
  - 25日MA が上向き（上昇トレンド確認）
  - 5日MA が 25日MA を下から上に抜けた（押し目からの回復）
  - 終値が 25日MA を下割れしたら撤退

Usage:
    cp .env.example .env  # 認証情報を設定
    uv run python backtest.py
    uv run python backtest.py --code 9984 --start 2022-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jquantsapi
import pandas as pd
from dotenv import load_dotenv
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

load_dotenv()

JST = timezone(timedelta(hours=9))
TSE = Venue("TSE")
JPY = Currency.from_str("JPY")


# ── J-Quants データ取得 ──────────────────────────────────────────────────────

def fetch_jquants(code: str, start: str, end: str) -> pd.DataFrame:
    """J-Quants から日足 OHLCV を取得する（調整済み価格）。"""
    api_key = os.environ["JQUANTS_API_KEY"]
    client = jquantsapi.ClientV2(api_key=api_key)
    # V2 API: from/to は YYYYMMDD 形式
    from_yyyymmdd = start.replace("-", "")
    to_yyyymmdd = end.replace("-", "")
    df = client.get_eq_bars_daily(code=code, from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd)
    df = df.dropna(subset=["AdjC"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  取得: {len(df)} 営業日 ({df['Date'].iloc[0]} 〜 {df['Date'].iloc[-1]})")
    return df


# ── NautilusTrader 型変換 ────────────────────────────────────────────────────

def make_instrument(code: str) -> Equity:
    """銘柄定義（日本株・調整済み価格 0.1円単位）。"""
    return Equity(
        instrument_id=InstrumentId(Symbol(code), TSE),
        raw_symbol=Symbol(code),
        currency=JPY,
        price_precision=1,
        price_increment=Price.from_str("0.1"),
        lot_size=Quantity.from_int(100),
        ts_event=0,
        ts_init=0,
    )


def df_to_bars(df: pd.DataFrame, instrument: Equity, bar_type: BarType) -> list[Bar]:
    """J-Quants DataFrame → NautilusTrader Bar リスト。引け時刻 15:30 JST を基準にする。"""
    bars = []
    for _, row in df.iterrows():
        date = datetime.strptime(row["Date"], "%Y-%m-%d")
        ts = int(date.replace(hour=15, minute=30, tzinfo=JST).timestamp() * 1_000_000_000)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(float(row["AdjO"])),
                high=instrument.make_price(float(row["AdjH"])),
                low=instrument.make_price(float(row["AdjL"])),
                close=instrument.make_price(float(row["AdjC"])),
                volume=Quantity.from_int(int(row["AdjVo"] or 0)),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return bars


# ── BNF スタイル戦略 ─────────────────────────────────────────────────────────

class BNFStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    fast_period: int = 5
    slow_period: int = 25
    trade_size: int = 100  # 株数（1単元）


class BNFStrategy(Strategy):
    """
    BNF(小手川隆)スタイルの押し目買い戦略。

    エントリー条件:
      - 25日MA が上向き（直近1本より上）
      - 5日MA が 25日MA を下から上に抜けた（ゴールデンクロス）

    エグジット条件:
      - 終値が 25日MA を下割れ
    """

    def __init__(self, config: BNFStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.fast_period = config.fast_period
        self.slow_period = config.slow_period
        self.trade_size = config.trade_size
        self._closes: list[float] = []

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        bar_type = BarType(
            instrument_id=self.instrument_id,
            bar_spec=BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
            aggregation_source=AggregationSource.EXTERNAL,
        )
        self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._closes.append(float(bar.close))
        needed = self.slow_period + 2
        if len(self._closes) < needed:
            return

        closes = self._closes
        fast = self.fast_period
        slow = self.slow_period

        fast_ma = sum(closes[-fast:]) / fast
        slow_ma = sum(closes[-slow:]) / slow
        fast_ma_prev = sum(closes[-fast - 1 : -1]) / fast
        slow_ma_prev = sum(closes[-slow - 1 : -1]) / slow

        has_position = bool(self.cache.positions_open(instrument_id=self.instrument_id))

        slow_up = slow_ma > slow_ma_prev
        golden_cross = fast_ma_prev <= slow_ma_prev and fast_ma > slow_ma

        if slow_up and golden_cross and not has_position:
            self._buy()
        elif float(bar.close) < slow_ma and has_position:
            self._sell()

    def _buy(self) -> None:
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(self.trade_size),
            )
        )

    def _sell(self) -> None:
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(self.trade_size),
            )
        )


# ── バックテスト実行 ─────────────────────────────────────────────────────────

def run(code: str, start: str, end: str) -> None:
    print(f"\n=== NautilusTrader × J-Quants バックテスト ===")
    print(f"  銘柄: {code}  期間: {start} 〜 {end}\n")

    print("▶ J-Quants からデータ取得中...")
    df = fetch_jquants(code, start, end)

    instrument = make_instrument(code)
    bar_type = BarType(
        instrument_id=instrument.id,
        bar_spec=BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    bars = df_to_bars(df, instrument, bar_type)
    print(f"  Bar 変換: {len(bars)} 本\n")

    print("▶ バックテスト実行中...")
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BNF-SAMPLE-001"),
            logging=LoggingConfig(bypass_logging=True),
        )
    )
    engine.add_venue(
        venue=TSE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=JPY,
        starting_balances=[Money(5_000_000, JPY)],  # 初期資金 500万円
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(BNFStrategy(BNFStrategyConfig(instrument_id=instrument.id)))
    engine.run()

    print("\n=== 結果 ===")
    print(engine.trader.generate_account_report(TSE))
    print(engine.trader.generate_positions_report())
    print(engine.trader.generate_order_fills_report())

    result = engine.get_result()
    print(f"\n最終損益: {result}")
    engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="NautilusTrader × J-Quants バックテスト")
    parser.add_argument("--code", default="72030", help="J-Quants 証券コード（例: 72030=トヨタ）")
    parser.add_argument("--start", default="2020-01-01", help="開始日 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="終了日 YYYY-MM-DD")
    args = parser.parse_args()
    run(args.code, args.start, args.end)


if __name__ == "__main__":
    main()
