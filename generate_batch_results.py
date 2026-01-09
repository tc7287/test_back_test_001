
import os
import sys
import itertools
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.data_loader import get_universe_data, KOSPI_UNIVERSE
from backtest.bb_rsi_strategy import BBRSIStrategy
from backtest.engine import run_backtest_for_ticker
from backtest.result_manager import ResultManager
from backtest.metrics import calculate_all_metrics

TICKER_NAME_MAP = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "373220.KS": "LG에너지솔루션",
    "207940.KS": "삼성바이오로직스",
    "005380.KS": "현대차",
    "006400.KS": "삼성SDI",
    "051910.KS": "LG화학",
    "000270.KS": "기아",
    "035420.KS": "NAVER",
    "005490.KS": "POSCO홀딩스",
    "035720.KS": "카카오",
    "068270.KS": "셀트리온",
    "028260.KS": "삼성물산",
    "105560.KS": "KB금융",
    "012330.KS": "현대모비스",
    "055550.KS": "신한지주",
    "066570.KS": "LG전자",
    "003670.KS": "포스코퓨처엠",
    "096770.KS": "SK이노베이션",
    "034730.KS": "SK",
}

def run_single_combo(args):
    """단일 파라미터 조합 실행"""
    strategy_class, params, data_dict, combo_id, start_date, end_date = args
    
    strategy = strategy_class(**params)
    all_trades = []
    ticker_results = {}
    
    for ticker, df in data_dict.items():
        ticker_name = TICKER_NAME_MAP.get(ticker, ticker)
        result_df, trades = run_backtest_for_ticker(ticker, df, strategy)
        
        trade_details = []
        for t in trades:
            # 내부 메트릭용 (date 키 필수)
            details = {
                'date': t['date'],
                'exit_date': t.get('exit_date'),
                'entry_price': t['entry_price'],
                'exit_price': t['exit_price'],
                'return': t['return'],
                'rationale': t.get('rationale', '')
            }
            # 저장용 (사용자 요청 한글 헤더)
            save_details = {
                '진입날짜': t['date'],
                '청산날짜': t.get('exit_date'),
                '진입가': t['entry_price'],
                '청산가': t['exit_price'],
                '수익률': t['return'],
                '매매근거': t.get('rationale', '')
            }
            trade_details.append(save_details)
            all_trades.append({**details, 'ticker': ticker})
            
        ticker_results[ticker_name] = {'trade_details': trade_details}

    # 모든 거래 내역 요약용 (renamed)
    all_trades_save = []
    for t in all_trades:
        all_trades_save.append({
            '진입날짜': t['date'],
            '청산날짜': t.get('exit_date'),
            '종목명': TICKER_NAME_MAP.get(t['ticker'], t['ticker']),
            '진입가': t['entry_price'],
            '청산가': t['exit_price'],
            '수익률': t['return'],
            '매매근거': t.get('rationale', '')
        })

    metrics = calculate_all_metrics(all_trades, start_date, end_date)
    metrics_dict = {
        'avg_return': metrics.avg_return_per_trade,
        'win_rate': metrics.win_rate,
        'total_return': metrics.total_return
    }
    
    return combo_id, params, metrics_dict, all_trades_save, ticker_results

def main():
    strategy_name = "BB-RSI 역추세 전략"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # 기존 결과 삭제 로직 주석 처리 (권한 문제/느림 방지)
    # if os.path.exists(os.path.join("Batch_Results", strategy_name)):
    #     import shutil
    #     shutil.rmtree(os.path.join("Batch_Results", strategy_name))

    print("데이터 로딩 중...")
    universe = KOSPI_UNIVERSE[:10]
    data_dict = get_universe_data(universe, start_date, end_date)
    
    param_grid = {
        'bb_period': range(10, 51, 5),
        'bb_std': [x * 0.5 for x in range(2, 7)],
        'rsi_period': range(7, 15, 1),
        'rsi_oversold': range(20, 41, 5),
        'rsi_overbought': range(60, 91, 5),
        'stop_loss': [x * 0.005 for x in range(2, 11)]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_combos = len(combinations)
    limit = 10 
    
    res_mgr = ResultManager()
    summary_index = []
    
    print(f"총 {total_combos}개의 조합 중 {limit}개를 계산합니다.")
    
    for i, params in enumerate(combinations):
        combo_id = i + 1
        if combo_id % 20 == 0:
            print(f"진행 중: {combo_id}/{limit}")
            
        _, _, metrics, all_trades_save, ticker_results = run_single_combo((
            BBRSIStrategy, params, data_dict, combo_id, start_date, end_date
        ))
        
        res_mgr.save_result(strategy_name, combo_id, params, metrics, all_trades_save, ticker_results)
        
        summary_index.append({
            'id': combo_id,
            'params': params,
            'metrics': metrics
        })
        
        if combo_id >= limit: break

    index_path = os.path.join("Batch_Results", strategy_name, "index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(summary_index, f, ensure_ascii=False, indent=2)
        
    print(f"백테스트 완료! {limit}개 조합 결과가 Batch_Results/{strategy_name} 에 저장되었습니다.")

if __name__ == "__main__":
    main()
