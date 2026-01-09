
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

class ResultManager:
    """
    백테스트 결과를 영구 저장하고 관리하는 클래스
    
    Structure:
    Batch_Results/
      [Strategy_Name]/
        파라미터조합#[N]/
          파라미터조합.json (또는 txt)
          metrics.json (평균수익률, 승률 등 요약)
          모든종목/
            trades.csv
          [Ticker1]/
            trades.csv
          ...
    """
    
    def __init__(self, base_dir: str = "Batch_Results"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _get_strategy_dir(self, strategy_name: str) -> str:
        path = os.path.join(self.base_dir, strategy_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_combo_dir(self, strategy_name: str, combo_id: int) -> str:
        strategy_dir = self._get_strategy_dir(strategy_name)
        combo_dir = os.path.join(strategy_dir, f"파라미터조합#{combo_id}")
        os.makedirs(combo_dir, exist_ok=True)
        return combo_dir

    def save_result(self, strategy_name: str, combo_id: int, params: Dict[str, Any], 
                    metrics: Dict[str, Any], all_trades: List[Dict], 
                    ticker_results: Dict[str, Dict]):
        """
        백테스트 결과를 구조화된 폴더에 저장
        """
        combo_dir = self._get_combo_dir(strategy_name, combo_id)
        
        # 1. 파라미터 정보 저장 (JSON 및 TXT)
        with open(os.path.join(combo_dir, "파라미터조합.json"), 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(combo_dir, "파라미터조합.txt"), 'w', encoding='utf-8') as f:
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            
        # 2. 요약 지표 저장 (평균수익률, 승률 등)
        with open(os.path.join(combo_dir, "metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
            
        # 3. 모든 종목 거래 내역 저장 (CSV)
        all_trades_dir = os.path.join(combo_dir, "모든종목")
        os.makedirs(all_trades_dir, exist_ok=True)
        if all_trades:
            df = pd.DataFrame(all_trades)
            # 매매근거 엔터 제거
            for col in ['매매근거', 'rationale']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)
            
            df.to_csv(
                os.path.join(all_trades_dir, "trades.csv"), 
                index=False, encoding='utf-8-sig'
            )
            
        # 4. 개별 종목별 거래 내역 저장 (CSV)
        for ticker_name, result in ticker_results.items():
            safe_name = "".join([c for c in ticker_name if c.isalnum() or c in (' ', '-', '_')]).strip()
            ticker_dir = os.path.join(combo_dir, safe_name)
            os.makedirs(ticker_dir, exist_ok=True)
            trades = result.get('trade_details', [])
            if trades:
                df = pd.DataFrame(trades)
                # 매매근거 엔터 제거
                for col in ['매매근거', 'rationale']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)
                
                df.to_csv(
                    os.path.join(ticker_dir, "trades.csv"), 
                    index=False, encoding='utf-8-sig'
                )

    def load_metrics(self, strategy_name: str, combo_id: int) -> Optional[Dict]:
        """특정 조합의 요약 지표 로드"""
        path = os.path.join(self.base_dir, strategy_name, f"파라미터조합#{combo_id}", "metrics.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def load_params(self, strategy_name: str, combo_id: int) -> Optional[Dict]:
        """특정 조합의 파라미터 로드"""
        path = os.path.join(self.base_dir, strategy_name, f"파라미터조합#{combo_id}", "파라미터조합.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def list_combos(self, strategy_name: str) -> List[Dict]:
        """전략의 모든 저장된 결과 목록 조회 (index.json이 있으면 그것 사용, 없으면 폴더 스캔)"""
        strategy_dir = os.path.join(self.base_dir, strategy_name)
        if not os.path.exists(strategy_dir):
            return []
            
        # 성능을 위해 index.json 사용 권장 (Batch 실행 시 생성됨)
        index_path = os.path.join(strategy_dir, "index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        # 수동 스캔
        combos = []
        for d in os.listdir(strategy_dir):
            if d.startswith("파라미터조합#"):
                try:
                    combo_id = int(d.split("#")[1])
                    metrics = self.load_metrics(strategy_name, combo_id)
                    params = self.load_params(strategy_name, combo_id)
                    if metrics and params:
                        combos.append({
                            'id': combo_id,
                            'params': params,
                            'metrics': metrics
                        })
                except:
                    continue
        return combos
