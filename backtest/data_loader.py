"""
데이터 로더 모듈
- yfinance를 통한 한국 주식 데이터 수집
- OHLCV 데이터 처리
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional


# KOSPI 대표 종목 리스트 (상위 20개)
KOSPI_UNIVERSE = [
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "373220.KS",  # LG에너지솔루션
    "207940.KS",  # 삼성바이오로직스
    "005380.KS",  # 현대차
    "006400.KS",  # 삼성SDI
    "051910.KS",  # LG화학
    "000270.KS",  # 기아
    "035420.KS",  # NAVER
    "005490.KS",  # POSCO홀딩스
    "035720.KS",  # 카카오
    "068270.KS",  # 셀트리온
    "028260.KS",  # 삼성물산
    "105560.KS",  # KB금융
    "012330.KS",  # 현대모비스
    "055550.KS",  # 신한지주
    "066570.KS",  # LG전자
    "003670.KS",  # 포스코퓨처엠
    "096770.KS",  # SK이노베이션
    "034730.KS",  # SK
]


def get_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    개별 종목의 OHLCV 데이터 조회
    
    Args:
        ticker: 종목 코드 (예: "005930.KS")
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
    
    Returns:
        OHLCV DataFrame 또는 None
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"[WARN] No data for {ticker}")
            return None
        
        # 컬럼명 표준화
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # 필요한 컬럼만 선택
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df['ticker'] = ticker
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return None


def get_universe_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    유니버스 전체 종목의 데이터 조회
    
    Args:
        tickers: 종목 코드 리스트
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        {ticker: DataFrame} 딕셔너리
    """
    data = {}
    
    for ticker in tickers:
        df = get_stock_data(ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            data[ticker] = df
            print(f"[OK] Loaded {ticker}: {len(df)} rows")
    
    print(f"\n총 {len(data)}개 종목 데이터 로드 완료")
    return data


def prepare_backtest_data(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    백테스트용 통합 데이터 준비
    
    Args:
        data: {ticker: DataFrame} 딕셔너리
    
    Returns:
        통합된 DataFrame (MultiIndex)
    """
    all_data = []
    
    for ticker, df in data.items():
        df = df.copy()
        df['ticker'] = ticker
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data)
    combined = combined.reset_index()
    combined = combined.rename(columns={'index': 'date', 'Date': 'date'})
    
    # date 컬럼을 timezone-naive로 변환
    if combined['date'].dt.tz is not None:
        combined['date'] = combined['date'].dt.tz_localize(None)
    
    return combined


if __name__ == "__main__":
    # 테스트
    print("=== 데이터 로더 테스트 ===")
    data = get_universe_data(
        KOSPI_UNIVERSE[:5],  # 상위 5개만 테스트
        "2024-01-01",
        "2024-12-27"
    )
    
    if data:
        combined = prepare_backtest_data(data)
        print(f"\n통합 데이터: {len(combined)} rows")
        print(combined.head())
