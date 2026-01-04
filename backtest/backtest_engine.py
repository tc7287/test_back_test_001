"""
백테스트 엔진

In-sample, Out-of-sample, Forward Test 통합 실행
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from backtest.data_loader import get_universe_data, prepare_backtest_data, KOSPI_UNIVERSE
from backtest.volatility_breakout import VolatilityBreakoutStrategy
from backtest.metrics import (
    calculate_all_metrics, 
    print_metrics_report, 
    check_pass_criteria,
    PerformanceMetrics
)
from backtest.risk_manager import (
    RiskManager, 
    RiskConfig, 
    apply_risk_management_to_trades
)


class BacktestEngine:
    """
    백테스트 엔진
    
    4단계 백테스트 프로세스 실행:
    1. 전략 정의
    2. In-sample 백테스트
    3. Out-of-sample 백테스트
    4. Forward Test
    """
    
    def __init__(
        self,
        strategy: VolatilityBreakoutStrategy,
        universe: List[str] = None,
        initial_capital: float = 10000000
    ):
        self.strategy = strategy
        self.universe = universe or KOSPI_UNIVERSE
        self.initial_capital = initial_capital
        
        # 기간 설정
        self.periods = {
            'IS': {
                'start': '2022-01-01',
                'end': '2022-12-31',
                'name': 'In-sample (2022)'
            },
            'OOS': {
                'start': '2023-01-01',
                'end': '2023-12-31',
                'name': 'Out-of-sample (2023)'
            },
            'FT': {
                'start': '2024-01-01',
                'end': '2024-12-27',
                'name': 'Forward Test (2024)'
            }
        }
        
        # 결과 저장
        self.results = {}
        self.data_cache = {}
    
    def load_data(self, phase: str) -> Dict[str, pd.DataFrame]:
        """데이터 로드 (캐싱)"""
        period = self.periods[phase]
        cache_key = f"{period['start']}_{period['end']}"
        
        if cache_key not in self.data_cache:
            print(f"\n[{phase}] 데이터 로드 중... ({period['start']} ~ {period['end']})")
            self.data_cache[cache_key] = get_universe_data(
                self.universe,
                period['start'],
                period['end']
            )
        
        return self.data_cache[cache_key]
    
    def run_phase(
        self,
        phase: str,
        apply_risk_management: bool = False
    ) -> Dict:
        """
        단일 백테스트 단계 실행
        
        Args:
            phase: 'IS', 'OOS', 'FT'
            apply_risk_management: 리스크 관리 적용 여부
        """
        period = self.periods[phase]
        print(f"\n{'='*60}")
        print(f"  {period['name']} 백테스트 시작")
        print(f"{'='*60}")
        
        # 데이터 로드
        data = self.load_data(phase)
        
        if not data:
            print(f"[ERROR] 데이터를 로드할 수 없습니다.")
            return {}
        
        # 백테스트 실행
        result = self.strategy.backtest_universe(data)
        
        # 거래 리스트
        trades = result['all_trades']
        
        # Forward Test는 리스크 관리 적용
        if apply_risk_management and trades:
            print("\n[리스크 관리 적용 중...]")
            print(f"  - 슬리피지: 0.1%")
            print(f"  - 수수료: 0.015% (편도)")
            trades = apply_risk_management_to_trades(
                trades,
                self.initial_capital,
                RiskConfig()
            )
        
        # 성능 지표 계산
        metrics = calculate_all_metrics(
            trades,
            period['start'],
            period['end'],
            self.initial_capital
        )
        
        # 결과 출력
        print_metrics_report(metrics, period['name'])
        
        # 통과 기준 검증
        if phase == 'IS':
            criteria = check_pass_criteria(metrics)
            self._print_pass_criteria(criteria, phase)
        elif phase == 'OOS':
            # OOS는 IS 대비 급격한 성능 저하 없는지 확인
            if 'IS' in self.results:
                self._compare_with_is(metrics)
        
        # 결과 저장
        self.results[phase] = {
            'metrics': metrics,
            'trades': trades,
            'result': result
        }
        
        return self.results[phase]
    
    def _print_pass_criteria(self, criteria: Dict, phase: str) -> None:
        """통과 기준 출력"""
        print(f"\n[{phase}] 통과 기준 검증:")
        print(f"  Sharpe > 1.0:    {'✓ PASS' if criteria['sharpe'] else '✗ FAIL'}")
        print(f"  MDD < 25%:       {'✓ PASS' if criteria['mdd'] else '✗ FAIL'}")
        print(f"  CAGR > 10%:      {'✓ PASS' if criteria['cagr'] else '✗ FAIL'}")
        print(f"  Win Rate > 50%:  {'✓ PASS' if criteria['win_rate'] else '✗ FAIL'}")
        print(f"  {'='*30}")
        print(f"  전체 결과:       {'✓ PASS' if criteria['overall'] else '✗ FAIL'}")
    
    def _compare_with_is(self, oos_metrics: PerformanceMetrics) -> None:
        """IS 대비 OOS 성능 비교"""
        is_metrics = self.results['IS']['metrics']
        
        print(f"\n[OOS vs IS 비교]")
        print(f"  Sharpe: {is_metrics.sharpe_ratio:.2f} → {oos_metrics.sharpe_ratio:.2f}")
        print(f"  CAGR:   {is_metrics.cagr:.1f}% → {oos_metrics.cagr:.1f}%")
        print(f"  MDD:    {is_metrics.mdd:.1f}% → {oos_metrics.mdd:.1f}%")
        print(f"  승률:   {is_metrics.win_rate:.1f}% → {oos_metrics.win_rate:.1f}%")
        
        # 급격한 성능 저하 체크
        sharpe_drop = (is_metrics.sharpe_ratio - oos_metrics.sharpe_ratio) / max(is_metrics.sharpe_ratio, 0.01)
        if sharpe_drop > 0.5:
            print(f"\n  ⚠️ 경고: Sharpe Ratio가 50% 이상 감소했습니다. 과적합 가능성이 있습니다.")
        else:
            print(f"\n  ✓ OOS 성능이 IS 대비 크게 저하되지 않았습니다.")
    
    def run_all(self) -> Dict:
        """
        전체 백테스트 실행
        
        1. In-sample (전략 개발)
        2. Out-of-sample (과적합 검증)
        3. Forward Test (실전 조건)
        """
        print("\n" + "="*60)
        print("  변동성 돌파 전략 백테스트")
        print("  " + "="*56)
        print(f"  전략: {self.strategy.name}")
        print(f"  K값: {self.strategy.k}")
        print(f"  유니버스: {len(self.universe)}개 종목")
        print(f"  초기 자본: {self.initial_capital:,.0f}원")
        print("="*60)
        
        # Step 1: In-sample
        self.run_phase('IS', apply_risk_management=False)
        
        # Step 2: Out-of-sample
        self.run_phase('OOS', apply_risk_management=False)
        
        # Step 3: Forward Test
        self.run_phase('FT', apply_risk_management=True)
        
        # 최종 요약
        self._print_final_summary()
        
        return self.results
    
    def _print_final_summary(self) -> None:
        """최종 요약 출력"""
        print("\n" + "="*60)
        print("  최종 백테스트 요약")
        print("="*60)
        
        for phase_key in ['IS', 'OOS', 'FT']:
            if phase_key not in self.results:
                continue
            
            period = self.periods[phase_key]
            metrics = self.results[phase_key]['metrics']
            
            print(f"\n  [{period['name']}]")
            print(f"    거래: {metrics.total_trades:,}회")
            print(f"    수익률: {metrics.total_return:+.1f}%")
            print(f"    CAGR: {metrics.cagr:+.1f}%")
            print(f"    Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"    MDD: {metrics.mdd:.1f}%")
            print(f"    승률: {metrics.win_rate:.1f}%")
        
        print("\n" + "="*60)


def main():
    """메인 실행 함수"""
    print("\n" + "#"*60)
    print("#  변동성 돌파 전략 백테스트 시스템")
    print("#"*60)
    
    # 전략 정의 (Step 1)
    print("\n[Step 1] 전략 정의")
    print("-" * 40)
    print("전략명: 변동성 돌파 전략 (Volatility Breakout)")
    print("매수 조건: 당일 시가 + (전일 고가 - 전일 저가) × K 돌파")
    print("매도 조건: 익일 시가 청산")
    print("K값: 0.5")
    print("-" * 40)
    
    strategy = VolatilityBreakoutStrategy(k=0.5)
    
    # 백테스트 엔진 초기화 (상위 10개 종목으로 테스트)
    engine = BacktestEngine(
        strategy=strategy,
        universe=KOSPI_UNIVERSE[:10],  # 상위 10개 종목
        initial_capital=10000000  # 1000만원
    )
    
    # 전체 백테스트 실행
    results = engine.run_all()
    
    print("\n\n백테스트 완료!")
    return results


if __name__ == "__main__":
    main()
