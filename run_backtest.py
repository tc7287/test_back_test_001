"""
백테스트 실행 스크립트

변동성 돌파 전략 4단계 백테스트:
1. 전략 정의
2. In-sample 백테스트 (2019-2021)
3. Out-of-sample 백테스트 (2022-2023)
4. Forward Test (2024)
"""

import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.backtest_engine import main


if __name__ == "__main__":
    results = main()
