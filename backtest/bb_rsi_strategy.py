"""
BB-RSI 역추세 전략 (Bollinger Bands + RSI Counter-trend Strategy)

전략 로직:
- 매수 조건: Close ≤ Lower Band AND RSI ≤ 30 (과매도)
- 매도 조건: Close ≥ Middle Band (중심선 도달) 또는 손절
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from backtest.base_strategy import BaseStrategy, StrategyParameter


class BBRSIStrategy(BaseStrategy):
    """
    BB-RSI 역추세 전략
    
    가격의 변동성(Bollinger Bands)과 과매도 지표(RSI)를 결합한
    평균 회귀(Mean Reversion) 전략
    """
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        stop_loss: float = 0.02,
        use_aggressive_exit: bool = False
    ):
        """
        Args:
            bb_period: Bollinger Bands 이동평균 기간
            bb_std: 표준편차 배수
            rsi_period: RSI 기간
            rsi_oversold: 과매도 기준
            rsi_overbought: 과매수 기준
            stop_loss: 손절 비율 (0.02 = 2%)
            use_aggressive_exit: 공격적 청산 사용 여부 (False: 중심선, True: 상단선)
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss = stop_loss
        self.use_aggressive_exit = use_aggressive_exit
    
    @property
    def name(self) -> str:
        return "BB-RSI 역추세 전략"
    
    @property
    def description(self) -> str:
        return "Bollinger Bands + RSI를 결합한 평균 회귀 전략"
    
    def get_parameters(self) -> List[StrategyParameter]:
        return [
            StrategyParameter(
                name="bb_period",
                label="BB 기간",
                default=20,
                min_value=10,
                max_value=50,
                step=5,
                param_type="int"
            ),
            StrategyParameter(
                name="bb_std",
                label="BB 표준편차",
                default=2.0,
                min_value=1.0,
                max_value=3.0,
                step=0.5,
                param_type="float"
            ),
            StrategyParameter(
                name="rsi_period",
                label="RSI 기간",
                default=14,
                min_value=7,
                max_value=14,
                step=1,
                param_type="int"
            ),
            StrategyParameter(
                name="rsi_oversold",
                label="RSI 과매도",
                default=30,
                min_value=20,
                max_value=40,
                step=5,
                param_type="int"
            ),
            StrategyParameter(
                name="rsi_overbought",
                label="RSI 과매수 (청산)",
                default=70,
                min_value=60,
                max_value=90,
                step=5,
                param_type="int"
            ),
            StrategyParameter(
                name="stop_loss",
                label="손절 비율 (%)",
                default=2.0,
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                param_type="float"
            ),
            StrategyParameter(
                name="use_aggressive_exit",
                label="공격적 익절 (상단선)",
                default=False,
                param_type="bool"
            )
        ]
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float):
        """Bollinger Bands 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return sma, upper_band, lower_band
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매수/매도 시그널 생성"""
        df = df.copy()
        
        # Bollinger Bands 계산
        df['bb_middle'], df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(
            df['close'], self.bb_period, self.bb_std
        )
        
        # RSI 계산
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # 매수 신호: Close ≤ Lower Band AND RSI ≤ 과매도
        df['buy_signal'] = (df['close'] <= df['bb_lower']) & (df['rsi'] <= self.rsi_oversold)
        
        # 매수가: 익일 시가 (시그널 발생 다음 날 진입)
        df['entry_price'] = np.where(
            df['buy_signal'],
            df['open'].shift(-1),
            np.nan
        )
        
        # 청산 조건 계산
        df['exit_price'] = np.nan
        df['exit_reason'] = ''
        
        in_position = False
        entry_price = 0
        entry_idx = None
        
        for i in range(len(df)):
            if df['buy_signal'].iloc[i] and not in_position:
                # 매수 진입 (익일 시가)
                if i + 1 < len(df):
                    in_position = True
                    entry_price = df['open'].iloc[i + 1]
                    entry_idx = i
            
            elif in_position and i > entry_idx:
                current_price = df['close'].iloc[i]
                
                # 익절 기준 설정 (상단선 vs 중심선)
                target_band = df['bb_upper'].iloc[i] if self.use_aggressive_exit else df['bb_middle'].iloc[i]
                
                # 손절 체크
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct <= -self.stop_loss:
                    df.loc[df.index[entry_idx], 'exit_price'] = current_price
                    df.loc[df.index[entry_idx], 'exit_date'] = df.index[i]
                    df.loc[df.index[entry_idx], 'exit_reason'] = 'stop_loss'
                    in_position = False
                
                # 익절 체크 1: 목표 밴드 도달
                elif current_price >= target_band:
                    df.loc[df.index[entry_idx], 'exit_price'] = current_price
                    df.loc[df.index[entry_idx], 'exit_date'] = df.index[i]
                    df.loc[df.index[entry_idx], 'exit_reason'] = 'take_profit_band'
                    in_position = False
                    
                # 익절 체크 2: RSI 과매수 도달
                elif df['rsi'].iloc[i] >= self.rsi_overbought:
                    df.loc[df.index[entry_idx], 'exit_price'] = current_price
                    df.loc[df.index[entry_idx], 'exit_date'] = df.index[i]
                    df.loc[df.index[entry_idx], 'exit_reason'] = 'take_profit_rsi'
                    in_position = False
        
        # 수익률 계산
        df['returns'] = np.where(
            df['buy_signal'] & df['exit_price'].notna(),
            (df['exit_price'] - df['entry_price']) / df['entry_price'],
            0
        )
        
        return df
    
    def get_trade_rationale(self, row: pd.Series, ticker: str) -> str:
        """거래 근거 생성"""
        date_str = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
        
        exit_reason_map = {
            'take_profit_band': f"목표 밴드 도달 ({'상단선' if self.use_aggressive_exit else '중심선'})",
            'take_profit_rsi': f"RSI 과매수 ({self.rsi_overbought} 이상)",
            'stop_loss': "손절매"
        }
        exit_reason = exit_reason_map.get(row.get('exit_reason'), "기타")
        
        return f"""
**매수 근거 (BB-RSI 역추세)**

📊 **시그널 발생일**: {date_str}

📉 **과매도 조건 충족**:
- Bollinger Lower Band: ₩{row['bb_lower']:,.0f}
- 종가: ₩{row['close']:,.0f}
- 종가 ≤ Lower Band: ✅

📈 **RSI 조건 충족**:
- RSI({self.rsi_period}): {row['rsi']:.1f}
- 과매도 기준: {self.rsi_oversold}
- RSI ≤ {self.rsi_oversold}: ✅

💰 **매매 결과**:
- 진입가 (익일 시가): ₩{row['entry_price']:,.0f}
- 청산가: ₩{row['exit_price']:,.0f}
- 청산 사유: {exit_reason}
- 수익률: {row['returns']*100:+.2f}%
"""

    def get_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """차트 표시 지표"""
        indicators = []
        
        # Bollinger Bands (Overlay)
        if 'bb_upper' in df.columns:
            indicators.append({
                'name': 'BB Upper', 
                'data': df['bb_upper'], 
                'type': 'overlay', 
                'color': 'rgba(100, 100, 255, 0.6)'
            })
            indicators.append({
                'name': 'BB Lower', 
                'data': df['bb_lower'], 
                'type': 'overlay', 
                'color': 'rgba(100, 100, 255, 0.6)'
            })
            indicators.append({
                'name': 'BB Middle', 
                'data': df['bb_middle'], 
                'type': 'overlay', 
                'color': 'rgba(255, 165, 0, 0.8)'
            })
            
        # Bollinger Bands (Secondary - Bandwidth)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
            bandwidth = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
            indicators.append({
                'name': 'BB Bandwidth',
                'data': bandwidth,
                'type': 'secondary',
                'color': '#4682B4' # SteelBlue
            })
            
        # RSI (Secondary)
        if 'rsi' in df.columns:
            indicators.append({
                'name': 'RSI', 
                'data': df['rsi'], 
                'type': 'secondary', 
                'color': '#9370DB', # MediumPurple
                'axis_range': [0, 100]
            })
            indicators.append({
                'name': 'Overbought', 
                'data': pd.Series([self.rsi_overbought]*len(df), index=df.index), 
                'type': 'secondary', 
                'color': 'red',
                'dash': 'dot'
            })
            indicators.append({
                'name': 'Oversold', 
                'data': pd.Series([self.rsi_oversold]*len(df), index=df.index), 
                'type': 'secondary', 
                'color': 'green',
                'dash': 'dot'
            })
            
        return indicators


if __name__ == "__main__":
    # 테스트
    print("=== BB-RSI 역추세 전략 테스트 ===")
    
    import numpy as np
    
    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 50000 + np.cumsum(np.random.randn(100) * 500)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 100,
        'high': prices + abs(np.random.randn(100) * 300),
        'low': prices - abs(np.random.randn(100) * 300),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    strategy = BBRSIStrategy()
    result = strategy.backtest_single(test_data, "TEST")
    
    print(f"전략: {strategy.name}")
    print(f"총 거래: {result['trades']}회")
    print(f"승률: {result['win_rate']:.1f}%")
    print(f"평균 수익률: {result['avg_return']:.2f}%")
