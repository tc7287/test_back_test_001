"""
리스크 관리 모듈 (Forward Test용)

- 거래당 리스크 제한
- 계좌 전체 리스크 제한
- 최대 보유 종목 수 제한
- 슬리피지/수수료 반영
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    risk_per_trade: float = 0.01  # 거래당 리스크 1%
    max_account_risk: float = 0.05  # 계좌 전체 리스크 5%
    max_positions: int = 10  # 최대 보유 종목 수
    slippage: float = 0.001  # 슬리피지 0.1%
    commission: float = 0.00015  # 수수료 0.015% (편도)


class RiskManager:
    """
    리스크 관리자
    
    Forward Test에서 실전과 동일한 조건으로 리스크 관리
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.current_positions: Dict[str, Dict] = {}
        self.account_equity = 0
    
    def set_initial_equity(self, equity: float) -> None:
        """초기 자본금 설정"""
        self.account_equity = equity
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float = None
    ) -> float:
        """
        포지션 크기 계산
        
        거래당 리스크 기준으로 진입 수량 결정
        """
        if self.account_equity <= 0:
            return 0
        
        # 거래당 최대 손실 금액
        max_loss_amount = self.account_equity * self.config.risk_per_trade
        
        # 손절가가 없으면 진입가의 5%를 손절로 가정
        if stop_loss_price is None:
            stop_loss_price = entry_price * 0.95
        
        # 주당 리스크
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # 진입 수량
        position_size = max_loss_amount / risk_per_share
        
        # 금액 기준 제한
        max_position_value = self.account_equity / self.config.max_positions
        max_shares_by_value = max_position_value / entry_price
        
        return min(position_size, max_shares_by_value)
    
    def can_open_position(self, ticker: str) -> bool:
        """새 포지션 진입 가능 여부"""
        # 이미 보유 중인지 확인
        if ticker in self.current_positions:
            return False
        
        # 최대 보유 종목 수 확인
        if len(self.current_positions) >= self.config.max_positions:
            return False
        
        # 계좌 전체 리스크 확인
        current_risk = self._calculate_current_risk()
        if current_risk >= self.config.max_account_risk:
            return False
        
        return True
    
    def _calculate_current_risk(self) -> float:
        """현재 계좌 전체 리스크 계산"""
        if not self.current_positions or self.account_equity <= 0:
            return 0
        
        total_risk = sum(
            pos.get('risk_amount', 0) 
            for pos in self.current_positions.values()
        )
        
        return total_risk / self.account_equity
    
    def apply_costs(
        self,
        entry_price: float,
        exit_price: float,
        is_entry: bool = True
    ) -> tuple:
        """
        슬리피지 및 수수료 적용
        
        Returns:
            (조정된 진입가, 조정된 청산가)
        """
        # 슬리피지: 매수 시 불리하게, 매도 시 불리하게
        adjusted_entry = entry_price * (1 + self.config.slippage)
        adjusted_exit = exit_price * (1 - self.config.slippage)
        
        # 수수료 (왕복)
        commission_cost = self.config.commission * 2  # 편도 × 2
        
        return adjusted_entry, adjusted_exit, commission_cost
    
    def calculate_adjusted_return(
        self,
        entry_price: float,
        exit_price: float
    ) -> float:
        """
        슬리피지/수수료 반영 수익률 계산
        """
        adj_entry, adj_exit, comm = self.apply_costs(entry_price, exit_price)
        
        raw_return = (adj_exit - adj_entry) / adj_entry
        adjusted_return = raw_return - comm
        
        return adjusted_return
    
    def open_position(
        self,
        ticker: str,
        entry_price: float,
        quantity: float,
        stop_loss: float = None
    ) -> bool:
        """포지션 진입"""
        if not self.can_open_position(ticker):
            return False
        
        if stop_loss is None:
            stop_loss = entry_price * 0.95
        
        risk_per_share = abs(entry_price - stop_loss)
        
        self.current_positions[ticker] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'risk_amount': risk_per_share * quantity
        }
        
        return True
    
    def close_position(self, ticker: str, exit_price: float) -> Optional[Dict]:
        """포지션 청산"""
        if ticker not in self.current_positions:
            return None
        
        pos = self.current_positions.pop(ticker)
        
        # 슬리피지/수수료 반영
        adj_return = self.calculate_adjusted_return(
            pos['entry_price'],
            exit_price
        )
        
        pnl = pos['entry_price'] * pos['quantity'] * adj_return
        
        self.account_equity += pnl
        
        return {
            'ticker': ticker,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'return': adj_return,
            'pnl': pnl
        }
    
    def get_summary(self) -> Dict:
        """리스크 관리 요약"""
        return {
            'equity': self.account_equity,
            'open_positions': len(self.current_positions),
            'current_risk': self._calculate_current_risk() * 100,
            'config': {
                'risk_per_trade': self.config.risk_per_trade * 100,
                'max_account_risk': self.config.max_account_risk * 100,
                'max_positions': self.config.max_positions,
                'slippage': self.config.slippage * 100,
                'commission': self.config.commission * 100
            }
        }


def apply_risk_management_to_trades(
    trades: List[Dict],
    initial_capital: float = 10000000,
    config: RiskConfig = None
) -> List[Dict]:
    """
    거래 리스트에 리스크 관리 적용
    
    Args:
        trades: 원본 거래 리스트
        initial_capital: 초기 자본금
        config: 리스크 설정
    
    Returns:
        리스크 관리가 적용된 거래 리스트
    """
    rm = RiskManager(config)
    rm.set_initial_equity(initial_capital)
    
    adjusted_trades = []
    
    for trade in sorted(trades, key=lambda x: x['date']):
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        
        if entry_price <= 0 or exit_price <= 0:
            continue
        
        # 슬리피지/수수료 반영 수익률
        adj_return = rm.calculate_adjusted_return(entry_price, exit_price)
        
        adjusted_trades.append({
            **trade,
            'original_return': trade.get('return', 0),
            'return': adj_return,
            'slippage_applied': True
        })
        
        # 자본금 업데이트 (단순화)
        rm.account_equity *= (1 + adj_return * (1 / rm.config.max_positions))
    
    return adjusted_trades


if __name__ == "__main__":
    print("=== 리스크 관리 테스트 ===")
    
    rm = RiskManager()
    rm.set_initial_equity(10000000)  # 1000만원
    
    print(f"초기 자본: {rm.account_equity:,.0f}원")
    
    # 포지션 크기 계산
    entry = 50000
    stop = 47500  # 5% 손절
    size = rm.calculate_position_size(entry, stop)
    print(f"진입가 {entry:,}원, 손절가 {stop:,}원")
    print(f"  → 적정 수량: {size:.1f}주")
    
    # 수익률 조정
    adj_return = rm.calculate_adjusted_return(50000, 52500)
    raw_return = (52500 - 50000) / 50000
    print(f"\n원본 수익률: {raw_return*100:.2f}%")
    print(f"조정 수익률: {adj_return*100:.2f}%")
    print(f"슬리피지/수수료 차감: {(raw_return - adj_return)*100:.2f}%")
