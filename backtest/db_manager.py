
import sqlite3
import pandas as pd
import os
from typing import Optional
from datetime import datetime

class DBManager:
    """
    시장 데이터를 SQLite DB에 저장하고 관리하는 클래스
    """
    
    def __init__(self, db_path: str = ".cache/market_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def _init_db(self):
        """DB 테이블 초기화"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # OHLCV 데이터 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # 인덱스 생성 (조회 속도 향상)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON market_data (ticker, date)")
        
        conn.commit()
        conn.close()
        
    def save_data(self, df: pd.DataFrame, ticker: str):
        """데이터프레임을 DB에 저장 (UPSERT)"""
        if df.empty:
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # DataFrame 전처리
        df = df.copy()
        if 'ticker' not in df.columns:
            df['ticker'] = ticker
            
        # 인덱스가 날짜인 경우 컬럼으로 변환
        if not 'date' in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'date'
            df = df.reset_index()
            
        # 날짜 포맷 통일 (strftime)
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
        # 데이터 삽입 (INSERT OR REPLACE)
        records = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        
        cursor.executemany("""
            INSERT OR REPLACE INTO market_data (ticker, date, open, high, low, close, volume)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        """, records)
        
        conn.commit()
        conn.close()
        print(f"[DB] Saved {len(records)} rows for {ticker}")

    def load_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """DB에서 데이터 조회"""
        conn = self._get_connection()
        
        query = """
            SELECT date, open, high, low, close, volume 
            FROM market_data 
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
            
            if not df.empty:
                # 날짜 컬럼을 datetime 객체로 변환하고 인덱스로 설정
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df
                
        except Exception as e:
            print(f"[DB Error] Failed to load data for {ticker}: {e}")
            
        finally:
            conn.close()
            
        return None
        
    def get_latest_date(self, ticker: str) -> Optional[str]:
        """해당 종목의 가장 최근 데이터 날짜 조회"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(date) FROM market_data WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
