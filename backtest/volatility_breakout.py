"""
변동성 돌파 전략 (Volatility Breakout Strategy)

전략 로직:
- 매수 조건: 당일 시가 + (전일 고가 - 전일 저가) × K > 현재가 돌파 시
- 매도 조건: 익일 시가 청산 (1일 홀딩)
- K값: 0.5 (기본값)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from backtest.base_strategy import BaseStrategy, StrategyParameter


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    변동성 돌파 전략
    
    Larry Williams의 변동성 돌파 전략을 일봉 기반으로 구현
    """
    
    def __init__(self, k: float = 0.5):
        """
        Args:
            k: 변동성 계수 (기본값 0.5)
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return "변동성 돌파 전략"
    
    @property
    def description(self) -> str:
        return "Larry Williams의 변동성 돌파 전략 (일봉 기반)"
    
    def get_parameters(self) -> List[StrategyParameter]:
        return [
            StrategyParameter(
                name="k",
                label="K값 (변동성 계수)",
                default=0.5,
                min_value=0.3,
                max_value=0.9,
                step=0.1,
                param_type="float"
            )
        ]
    
    def calculate_target_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        목표가 계산
        
        목표가 = 당일 시가 + (전일 고가 - 전일 저가) × K
        """
        df = df.copy()
        
        # 전일 변동폭
        df['prev_range'] = (df['high'].shift(1) - df['low'].shift(1))
        
        # 목표가 = 당일 시가 + 전일 변동폭 × K
        df['target_price'] = df['open'] + df['prev_range'] * self.k
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        매수/매도 신호 생성
        
        Args:
            df: 단일 종목 OHLCV 데이터 (date, open, high, low, close, volume)
        
        Returns:
            신호가 추가된 DataFrame
        """
        df = self.calculate_target_price(df.copy())
        
        # 매수 신호: 고가가 목표가를 돌파한 경우
        df['buy_signal'] = df['high'] >= df['target_price']
        
        # 매수가: 목표가 (돌파 시점 가격)
        df['entry_price'] = np.where(
            df['buy_signal'],
            df['target_price'],
            np.nan
        )
        
        # 매도가: 익일 시가 (다음날 시가 청산)
        df['exit_price'] = df['open'].shift(-1)
        
        # 청산 날짜: 다음날
        df['exit_date'] = df.index.shift(1, freq='D') # 이렇게 하면 영업일 아닐 수 있음.
        # 정확히는 df['open'].shift(-1)을 가져온 행의 날짜...가 아니라,
        # entry_price가 설정된 날(오늘) -> exit_price가 설정된 날(다음날)
        # df['exit_date'] = df.index로 하고 shift(-1)?
        # index shift는 freq 필요.
        # 가장 정확한건: df['date'].shift(-1)
        # 하지만 index가 date임.
        # df.index.to_series().shift(-1)
        
        # 인덱스(날짜)를 컬럼으로 빼서 shift
        dates = df.index.to_series()
        df['exit_date'] = dates.shift(-1)

        # 수익률 계산
        df['returns'] = np.where(
            df['buy_signal'],
            (df['exit_price'] - df['entry_price']) / df['entry_price'],
            0
        )
        
        return df
        
    def get_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        indicators = []
        if 'target_price' in df.columns:
            indicators.append({
                'name': 'Target Price',
                'data': df['target_price'],
                'type': 'overlay',
                'color': 'rgba(255, 165, 0, 0.7)',
                'dash': 'dot'
            })
        return indicators
    
    def get_trade_rationale(self, row: pd.Series, ticker: str) -> str:
        """거래 근거 생성"""
        date_str = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
        
        return f"""
**매수 근거 (변동성 돌파)**

📊 **시그널 발생일**: {date_str}

📈 **돌파 조건**:
- 전일 변동폭: ₩{row['prev_range']:,.0f}
- K값: {self.k}
- 목표가 = 당일 시가(₩{row['open']:,.0f}) + 변동폭 × K
- **목표가**: ₩{row['target_price']:,.0f}

✅ **진입 조건 충족**:
- 당일 고가(₩{row['high']:,.0f}) ≥ 목표가(₩{row['target_price']:,.0f})
- 돌파 확인 → **매수 실행**

💰 **매매 결과**:
- 진입가: ₩{row['entry_price']:,.0f}
- 청산가(익일 시가): ₩{row['exit_price']:,.0f}
- 수익률: {row['returns']*100:+.2f}%
"""
    
    # 기존 호환성을 위해 backtest_single, backtest_universe는 BaseStrategy에서 상속


if __name__ == "__main__":
    # 테스트용 더미 데이터
    print("=== 변동성 돌파 전략 테스트 ===")
    
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    test_data = pd.DataFrame({
        'open': [100, 102, 101, 105, 103, 108, 106, 110, 108, 112],
        'high': [103, 104, 106, 107, 109, 110, 112, 113, 115, 114],
        'low': [99, 100, 100, 103, 102, 106, 105, 108, 107, 110],
        'close': [102, 101, 105, 103, 108, 106, 110, 108, 112, 111],
        'volume': [1000] * 10
    }, index=dates)
    
    strategy = VolatilityBreakoutStrategy(k=0.5)
    result = strategy.backtest_single(test_data, "TEST")
    
    print(f"전략: {strategy.name}")
    print(f"총 거래: {result['trades']}회")
    print(f"승률: {result['win_rate']:.1f}%")
    print(f"평균 수익률: {result['avg_return']:.2f}%")
