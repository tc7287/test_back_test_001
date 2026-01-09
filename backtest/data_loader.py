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


from backtest.db_manager import DBManager

# DB 매니저 인스턴스
db_manager = DBManager()

def get_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    개별 종목의 OHLCV 데이터 조회 (DB 캐싱 적용)
    """
    # 1. DB에서 데이터 조회
    df = db_manager.load_data(ticker, start_date, end_date)
    
    # 2. 데이터가 없거나 최신 데이터가 필요한지 확인
    need_update = False
    fetch_start = start_date
    
    if df is None or df.empty:
        need_update = True
    else:
        # 마지막 데이터 날짜 확인
        last_date = df.index[-1]
        target_end_date = pd.to_datetime(end_date)
        
        # 오늘 날짜 (미래 데이터 요청 방지)
        today = pd.Timestamp.now().normalize()
        if target_end_date > today:
            target_end_date = today
            
        # 마지막 데이터가 목표 종료일보다 3일 이상 오래되었으면 업데이트 시도 (주말/휴장일 고려 여유)
        if last_date < target_end_date - pd.Timedelta(days=3):
            need_update = True
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # 3. 필요 시 yfinance에서 다운로드 및 DB 저장
    if need_update:
        try:
            print(f"[FETCH] Downloading {ticker} from {fetch_start} to {end_date}")
            stock = yf.Ticker(ticker)
            new_data = stock.history(start=fetch_start, end=end_date)
            
            if not new_data.empty:
                # 컬럼명 표준화
                new_data = new_data.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # 필요한 컬럼만 선택
                new_data = new_data[['open', 'high', 'low', 'close', 'volume']]
                new_data['ticker'] = ticker
                
                # DB 저장 (Upsert)
                db_manager.save_data(new_data, ticker)
                
                # 데이터 다시 로드 (병합 효과)
                df = db_manager.load_data(ticker, start_date, end_date)
            elif df is None:
                print(f"[WARN] No data found for {ticker}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker}: {e}")
            # 에러 발생 시 기존 데이터라도 반환
            if df is not None:
                return df
            return None
            
    return df


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
