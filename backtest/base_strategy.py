"""
전략 기본 클래스 (Base Strategy)

모든 백테스트 전략의 기본 인터페이스 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass
class StrategyParameter:
    """전략 파라미터 정의"""
    name: str
    label: str
    default: Any
    min_value: Any = None
    max_value: Any = None
    step: Any = None
    param_type: str = "float"  # float, int, bool


class BaseStrategy(ABC):
    """
    전략 기본 추상 클래스
    
    모든 전략은 이 클래스를 상속하여 구현
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """전략 설명"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[StrategyParameter]:
        """
        조절 가능한 파라미터 목록 반환
        
        Returns:
            StrategyParameter 리스트
        """
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        매수/매도 시그널 생성
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            시그널이 추가된 DataFrame
        """
        pass
    
    @abstractmethod
    def get_trade_rationale(self, row: pd.Series, ticker: str) -> str:
        """
        거래 근거 생성
        
        Args:
            row: 거래 시점의 데이터
            ticker: 종목 코드
        
        Returns:
            매매 근거 문자열 (마크다운)
        """
        pass
    
    def backtest_single(self, df: pd.DataFrame, ticker: str) -> Dict:
        """
        단일 종목 백테스트
        
        Args:
            df: 종목 데이터
            ticker: 종목 코드
        
        Returns:
            백테스트 결과 딕셔너리
        """
        result_df = self.generate_signals(df)
        
        # 유효한 거래만 필터링
        trades = result_df[
            (result_df['buy_signal']) & 
            (result_df['exit_price'].notna())
        ].copy()
        
        if len(trades) == 0:
            return {
                'ticker': ticker,
                'trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'trade_details': [],
                'result_df': result_df
            }
        
        # 거래 통계
        wins = trades[trades['returns'] > 0]
        losses = trades[trades['returns'] <= 0]
        
        trade_details = []
        for idx, row in trades.iterrows():
            trade_details.append({
                'date': idx if isinstance(idx, pd.Timestamp) else row.get('date', idx),
                'entry_price': row['entry_price'],
                'exit_price': row['exit_price'],
                'return': row['returns'],
                'rationale': self.get_trade_rationale(row, ticker)
            })
        
        return {
            'ticker': ticker,
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if len(trades) > 0 else 0,
            'avg_return': trades['returns'].mean() * 100,
            'total_return': (1 + trades['returns']).prod() - 1,
            'trade_details': trade_details,
            'result_df': result_df
        }
    
    def backtest_universe(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        유니버스 전체 백테스트
        
        Args:
            data: {ticker: DataFrame} 딕셔너리
        
        Returns:
            전체 백테스트 결과
        """
        import numpy as np
        
        results = {}
        all_trades = []
        
        for ticker, df in data.items():
            result = self.backtest_single(df, ticker)
            results[ticker] = result
            all_trades.extend(result['trade_details'])
        
        # 전체 통계
        total_trades = sum(r['trades'] for r in results.values())
        total_wins = sum(r.get('wins', 0) for r in results.values())
        
        if total_trades > 0:
            all_returns = [t['return'] for t in all_trades]
            avg_return = np.mean(all_returns) * 100
            overall_win_rate = total_wins / total_trades * 100
        else:
            avg_return = 0
            overall_win_rate = 0
        
        return {
            'strategy': self.name,
            'total_trades': total_trades,
            'total_wins': total_wins,
            'overall_win_rate': overall_win_rate,
            'avg_return_per_trade': avg_return,
            'results_by_ticker': results,
            'all_trades': all_trades
        }
