"""
성능 지표 계산 모듈

- Sharpe Ratio
- CAGR (연간 복리 수익률)
- MDD (최대 낙폭)
- Win Rate (승률)
- 손익비 (Profit Factor)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """백테스트 성능 지표"""
    sharpe_ratio: float
    cagr: float
    mdd: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    total_return: float
    avg_return_per_trade: float


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    샤프 비율 계산
    
    Sharpe = (평균 수익률 - 무위험 수익률) / 수익률 표준편차
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    return sharpe


def calculate_cagr(
    total_return: float,
    years: float
) -> float:
    """
    CAGR (연간 복리 수익률) 계산
    
    CAGR = (1 + 총수익률) ^ (1/년수) - 1
    """
    if years <= 0:
        return 0.0
    
    if total_return <= -1:
        return -1.0
    
    cagr = (1 + total_return) ** (1 / years) - 1
    return cagr


def calculate_mdd(equity_curve: pd.Series) -> float:
    """
    MDD (최대 낙폭) 계산
    
    MDD = max(고점 대비 하락률)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # 누적 고점
    running_max = equity_curve.cummax()
    
    # 낙폭
    drawdown = (equity_curve - running_max) / running_max
    
    # 최대 낙폭
    mdd = drawdown.min()
    
    return abs(mdd)


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    승률 계산
    
    승률 = 수익 거래 수 / 전체 거래 수
    """
    if len(trades) == 0:
        return 0.0
    
    wins = sum(1 for t in trades if t.get('return', 0) > 0)
    return wins / len(trades) * 100


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    손익비 (Profit Factor) 계산
    
    손익비 = 총 이익 / 총 손실
    """
    if len(trades) == 0:
        return 0.0
    
    profits = sum(t['return'] for t in trades if t.get('return', 0) > 0)
    losses = abs(sum(t['return'] for t in trades if t.get('return', 0) < 0))
    
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    
    return profits / losses


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    기대값 계산
    
    기대값 = (승률 × 평균이익) - (패률 × 평균손실)
    """
    if len(trades) == 0:
        return 0.0
    
    wins = [t['return'] for t in trades if t.get('return', 0) > 0]
    losses = [abs(t['return']) for t in trades if t.get('return', 0) < 0]
    
    win_rate = len(wins) / len(trades)
    loss_rate = len(losses) / len(trades)
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    return expectancy


def build_equity_curve(
    trades: List[Dict],
    initial_capital: float = 10000000
) -> pd.Series:
    """
    자산 곡선 생성
    
    Args:
        trades: 거래 리스트 [{'date': ..., 'return': ...}, ...]
        initial_capital: 초기 자본금
    
    Returns:
        자산 곡선 Series
    """
    if not trades:
        return pd.Series([initial_capital])
    
    # 날짜순 정렬
    sorted_trades = sorted(trades, key=lambda x: x['date'])
    
    equity = [initial_capital]
    for trade in sorted_trades:
        new_equity = equity[-1] * (1 + trade['return'])
        equity.append(new_equity)
    
    return pd.Series(equity)


def calculate_all_metrics(
    trades: List[Dict],
    start_date: str,
    end_date: str,
    initial_capital: float = 10000000
) -> PerformanceMetrics:
    """
    모든 성능 지표 계산
    
    Args:
        trades: 거래 리스트
        start_date: 시작일
        end_date: 종료일
        initial_capital: 초기 자본금
    
    Returns:
        PerformanceMetrics 객체
    """
    if not trades:
        return PerformanceMetrics(
            sharpe_ratio=0,
            cagr=0,
            mdd=0,
            win_rate=0,
            profit_factor=0,
            expectancy=0,
            total_trades=0,
            total_return=0,
            avg_return_per_trade=0
        )
    
    # 자산 곡선
    equity_curve = build_equity_curve(trades, initial_capital)
    
    # 기간 계산
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    years = (end - start).days / 365.25
    
    # 총 수익률
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    
    # 일별 수익률
    returns = pd.Series([t['return'] for t in trades])
    
    return PerformanceMetrics(
        sharpe_ratio=calculate_sharpe_ratio(returns),
        cagr=calculate_cagr(total_return, years) * 100,
        mdd=calculate_mdd(equity_curve) * 100,
        win_rate=calculate_win_rate(trades),
        profit_factor=calculate_profit_factor(trades),
        expectancy=calculate_expectancy(trades) * 100,
        total_trades=len(trades),
        total_return=total_return * 100,
        avg_return_per_trade=returns.mean() * 100 if len(returns) > 0 else 0
    )


def print_metrics_report(
    metrics: PerformanceMetrics,
    phase_name: str = "Backtest"
) -> None:
    """성능 지표 리포트 출력"""
    print(f"\n{'='*50}")
    print(f"  {phase_name} 결과")
    print(f"{'='*50}")
    print(f"  총 거래 횟수:      {metrics.total_trades:,}회")
    print(f"  총 수익률:         {metrics.total_return:+.2f}%")
    print(f"  CAGR:              {metrics.cagr:+.2f}%")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"  MDD:               {metrics.mdd:.2f}%")
    print(f"  승률:              {metrics.win_rate:.1f}%")
    print(f"  손익비:            {metrics.profit_factor:.2f}")
    print(f"  기대값:            {metrics.expectancy:+.2f}%")
    print(f"  평균 수익률/거래:  {metrics.avg_return_per_trade:+.2f}%")
    print(f"{'='*50}")


def check_pass_criteria(
    metrics: PerformanceMetrics,
    min_sharpe: float = 1.0,
    max_mdd: float = 25.0,
    min_cagr: float = 10.0,
    min_win_rate: float = 50.0
) -> Dict[str, bool]:
    """
    통과 기준 검증
    
    Returns:
        각 기준별 통과 여부 딕셔너리
    """
    results = {
        'sharpe': metrics.sharpe_ratio >= min_sharpe,
        'mdd': metrics.mdd <= max_mdd,
        'cagr': metrics.cagr >= min_cagr,
        'win_rate': metrics.win_rate >= min_win_rate or metrics.expectancy > 0,
        'overall': False
    }
    
    # 전체 통과 여부
    results['overall'] = all([
        results['sharpe'],
        results['mdd'],
        results['cagr'],
        results['win_rate']
    ])
    
    return results


if __name__ == "__main__":
    # 테스트
    print("=== 성능 지표 테스트 ===")
    
    test_trades = [
        {'date': '2024-01-01', 'return': 0.02},
        {'date': '2024-01-02', 'return': -0.01},
        {'date': '2024-01-03', 'return': 0.03},
        {'date': '2024-01-04', 'return': 0.01},
        {'date': '2024-01-05', 'return': -0.005},
    ]
    
    metrics = calculate_all_metrics(
        test_trades,
        '2024-01-01',
        '2024-01-05'
    )
    
    print_metrics_report(metrics, "테스트")
