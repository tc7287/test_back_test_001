import pandas as pd
from typing import Dict, List, Tuple, Any
from backtest.base_strategy import BaseStrategy
from backtest.metrics import calculate_all_metrics, PerformanceMetrics
from backtest.risk_manager import RiskManager, RiskConfig
# from backtest.cache_manager import BacktestCache  # 캐시 시스템 삭제

def run_backtest_for_ticker(ticker: str, df: pd.DataFrame, strategy: BaseStrategy) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    단일 종목 백테스트 실행 및 상세 정보 반환 (모든 전략 호환)
    """
    result_df = strategy.generate_signals(df.copy())
    
    trades = []
    for idx, row in result_df.iterrows():
        if row.get('buy_signal', False) and pd.notna(row.get('exit_price')):
            # row를 Series로 변환하여 name 속성 설정
            row_series = row.copy()
            row_series.name = idx
            
            trade = {
                'date': idx,
                'ticker': ticker,
                'type': 'BUY',
                'entry_price': row['entry_price'],
                'exit_price': row['exit_price'],
                'exit_date': row.get('exit_date'), # 청산 날짜 추가
                'return': row['returns'],
                # 전략의 get_trade_rationale 메서드 사용
                'rationale': strategy.get_trade_rationale(row_series, ticker)
            }
            trades.append(trade)
    
    return result_df, trades

def run_universe_backtest(
    strategy: BaseStrategy,
    data_dict: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    phase_name: str
) -> Tuple[PerformanceMetrics, List[Dict]]:
    """
    유니버스 전체 백테스트 실행 (캐싱 적용)
    
    Args:
        strategy: 전략 인스턴스
        data_dict: {ticker: DataFrame}
        start_date: 시작일
        end_date: 종료일
        phase_name: 단계 이름 (IS, OOS, FT)
        
    Returns:
        (전체 성능 지표, 전체 거래 리스트)
    """
    tickers = list(data_dict.keys())
    
    # 3. 백테스트 실행 (실시간 계산)
    all_trades = []
    results_by_ticker = {}
    
    for ticker, ticker_df in data_dict.items():
        _, ticker_trades = run_backtest_for_ticker(ticker, ticker_df, strategy)
        
        # 거래 정보 간소화하여 수집 (메모리 절약)
        for t in ticker_trades:
            all_trades.append({
                'date': t['date'],
                'return': t['return'],
                'entry_price': t['entry_price'],
                'exit_price': t['exit_price'],
                'ticker': ticker, # 티커 정보 추가
                'rationale': t.get('rationale', '') # 자세한 정보 필요 시
            })
            
    # FT(Forward Test) 단계일 경우 리스크 관리 적용
    if phase_name == 'FT' and all_trades:
        # RiskManager를 사용하여 슬리피지/수수료 등 적용
        # 여기서는 간단하게 기존 로직을 따름 (import risk_manager 필요)
        rm = RiskManager(RiskConfig())
        adjusted_trades = []
        for t in all_trades:
            # 원본 수익률 보존, 조정된 수익률 계산
            adj_return = rm.calculate_adjusted_return(t['entry_price'], t['exit_price'])
            t_copy = t.copy()
            t_copy['return'] = adj_return
            t_copy['original_return'] = t['return']
            adjusted_trades.append(t_copy)
        all_trades = adjusted_trades

    # 4. 메트릭 계산
    metrics = calculate_all_metrics(
        all_trades, start_date, end_date, 10000000
    )
    
    return metrics, all_trades
