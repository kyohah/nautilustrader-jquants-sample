"""
NautilusTrader × J-Quants バックテストサンプル

BNF(小手川隆)スタイルの乖離率逆張り戦略:
  エントリー条件（3つ複合）:
    1. 25日MA からの下方乖離率 -20% 以上（BNF 本人発言）
    2. 出来高が 20日平均の 3倍以上（セリクラ確認）
    3. ボリンジャーバンド -2σ 以下

  エグジット条件:
    - 乖離率が -2% 以上（25日MA への回帰）

Usage:
    cp .env.example .env  # 認証情報を設定
    uv run python backtest.py
    uv run python backtest.py --code 9984 --start 2020-01-01 --end 2024-12-31
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

import math


class BNFStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    ma_period: int = 25           # 25日MA（BNF 本人発言）
    entry_deviation: float = -20.0  # 乖離率エントリー閾値（%）
    exit_deviation: float = -2.0    # 乖離率エグジット閾値（%、MAへの回帰）
    bb_sigma: float = 2.0           # BB 倍率
    volume_ratio: float = 3.0       # 出来高急増判定（N日平均の何倍）
    volume_period: int = 20         # 出来高平均の算出期間
    trade_size: int = 100           # 株数（1単元）


class BNFStrategy(Strategy):
    """
    BNF(小手川隆)の乖離率逆張り戦略。

    BNF 本人発言:
      「25日移動平均線からのマイナス乖離が最低20%、
        安心して買えるのは35%以上の乖離率という感じだった」

    エントリー（3条件複合）:
      1. 25日MA 乖離率 <= -20%
      2. 出来高が 20日平均の 3倍以上（セリクラ確認）
      3. ボリンジャーバンド -2σ 以下

    エグジット:
      乖離率 >= -2%（25日MA への回帰）
    """

    def __init__(self, config: BNFStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.ma_period = config.ma_period
        self.entry_deviation = config.entry_deviation
        self.exit_deviation = config.exit_deviation
        self.bb_sigma = config.bb_sigma
        self.volume_ratio = config.volume_ratio
        self.volume_period = config.volume_period
        self.trade_size = config.trade_size
        self._closes: list[float] = []
        self._volumes: list[float] = []

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        bar_type = BarType(
            instrument_id=self.instrument_id,
            bar_spec=BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
            aggregation_source=AggregationSource.EXTERNAL,
        )
        self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        close = float(bar.close)
        volume = float(bar.volume)
        self._closes.append(close)
        self._volumes.append(volume)

        needed = max(self.ma_period, self.volume_period) + 1
        if len(self._closes) < needed:
            return

        # MA は現在バーを除く直近 ma_period 本で計算（標準・デバッグスクリプトと一致）
        ma25 = sum(self._closes[-self.ma_period - 1:-1]) / self.ma_period
        deviation = (close - ma25) / ma25 * 100

        # ボリンジャーバンド -Nσ（同じ window）
        variance = sum((c - ma25) ** 2 for c in self._closes[-self.ma_period - 1:-1]) / self.ma_period
        bb_lower = ma25 - self.bb_sigma * math.sqrt(variance)

        # 出来高急増（現在バーを除く直近 volume_period 日平均）
        avg_volume = sum(self._volumes[-self.volume_period - 1:-1]) / self.volume_period
        volume_surge = avg_volume > 0 and volume >= avg_volume * self.volume_ratio

        has_position = bool(self.cache.positions_open(instrument_id=self.instrument_id))

        if not has_position:
            if deviation <= self.entry_deviation and close <= bb_lower and volume_surge:
                self._buy(deviation, bb_lower, volume / avg_volume if avg_volume > 0 else 0)
        else:
            if deviation >= self.exit_deviation:
                self._sell(deviation)

    def _buy(self, deviation: float, bb_lower: float, vol_ratio: float) -> None:
        self.log.info(
            f"BUY  乖離率={deviation:.1f}%  BB-lower={bb_lower:.1f}  出来高比={vol_ratio:.1f}x"
        )
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(self.trade_size),
            )
        )

    def _sell(self, deviation: float) -> None:
        self.log.info(f"SELL 乖離率={deviation:.1f}%（MA 回帰）")
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
    engine.add_strategy(
        BNFStrategy(
            BNFStrategyConfig(
                instrument_id=instrument.id,
                entry_deviation=-15.0,  # 原典は -20% だが現代相場では -15% が現実的
                exit_deviation=-2.0,
            )
        )
    )
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
