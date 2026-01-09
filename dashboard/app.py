"""
백테스트 대시보드 (Multi-Strategy)

Streamlit 기반 인터랙티브 대시보드:
- 다중 전략 선택 (변동성돌파, BB-RSI 등)
- 종목별 차트 (OHLC + 매수/매도 시그널)
- 마커 클릭 시 거래 근거 표시
- 성능 지표 요약
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import get_universe_data, KOSPI_UNIVERSE
from backtest.volatility_breakout import VolatilityBreakoutStrategy
from backtest.bb_rsi_strategy import BBRSIStrategy
from backtest.metrics import calculate_all_metrics, PerformanceMetrics
from backtest.risk_manager import apply_risk_management_to_trades, RiskConfig
from backtest.engine import run_universe_backtest, run_backtest_for_ticker
from backtest.result_manager import ResultManager
from dashboard.chart_utils import create_advanced_chart

# 영구 저장소 관리자
res_mgr = ResultManager()

# 사용 가능한 전략 목록
AVAILABLE_STRATEGIES = {
    "변동성 돌파 전략": VolatilityBreakoutStrategy,
    "BB-RSI 역추세 전략": BBRSIStrategy,
}

# 페이지 설정
st.set_page_config(
    page_title="백테스트 대시보드",
    page_icon="📈",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
    }
    .phase-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .trade-detail {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .buy-signal { color: #22c55e; font-weight: bold; }
    .sell-signal { color: #ef4444; font-weight: bold; }
    .trade-history-container {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'zoom_date' not in st.session_state:
    st.session_state.zoom_date = None
if 'zoom_phase' not in st.session_state:
    st.session_state.zoom_phase = None


@st.cache_data(ttl=3600)
def load_all_data():
    """모든 기간 데이터 로드 (캐싱)"""
    periods = {
        'IS': ('2022-01-01', '2022-12-31', 'In-sample (2022)'),
        'OOS': ('2023-01-01', '2023-12-31', 'Out-of-sample (2023)'),
        'FT': ('2024-01-01', '2024-12-27', 'Forward Test (2024)')
    }
    
    all_data = {}
    for phase, (start, end, name) in periods.items():
        data = get_universe_data(KOSPI_UNIVERSE[:10], start, end)
        all_data[phase] = {
            'data': data,
            'start': start,
            'end': end,
            'name': name
        }
    
    return all_data








def display_metrics(metrics: PerformanceMetrics, phase_name: str):
    """성능 지표 표시"""
    cols = st.columns(4)
    
    with cols[0]:
        delta_color = "normal" if metrics.avg_return_per_trade >= 0 else "inverse"
        st.metric("평균 수익률", f"{metrics.avg_return_per_trade:+.2f}%", 
                  # (전체 수익률 / 총 거래 수) 임을 명시
                  delta=f"Total {metrics.total_return:+.1f}%", delta_color=delta_color)
    
    with cols[1]:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}",
                  delta="PASS ✓" if metrics.sharpe_ratio >= 1.0 else "FAIL ✗")
    
    with cols[2]:
        st.metric("MDD", f"{metrics.mdd:.1f}%",
                  delta="PASS ✓" if metrics.mdd <= 25 else "FAIL ✗",
                  delta_color="inverse" if metrics.mdd > 25 else "normal")
    
    with cols[3]:
        st.metric("승률", f"{metrics.win_rate:.1f}%",
                  delta=f"{metrics.total_trades}회 거래")


def main():
    st.markdown('<h1 class="main-header">📈 백테스트 대시보드</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 전략 선택 드롭다운
        st.markdown("### 📋 전략 선택")
        strategy_name = st.selectbox(
            "전략",
            list(AVAILABLE_STRATEGIES.keys()),
            key="strategy_select"
        )
        
        # 선택된 전략 클래스 가져오기
        StrategyClass = AVAILABLE_STRATEGIES[strategy_name]
        
        # 전략별 파라미터 동적 생성
        st.markdown("### 🔧 파라미터")
        
        # 임시 전략 인스턴스로 파라미터 목록 가져오기
        temp_strategy = StrategyClass()
        params = temp_strategy.get_parameters()
        
        param_values = {}
        for param in params:
            if param.param_type == "float":
                param_values[param.name] = st.slider(
                    param.label,
                    float(param.min_value),
                    float(param.max_value),
                    float(param.default),
                    float(param.step),
                    key=f"param_{param.name}"
                )
            elif param.param_type == "int":
                param_values[param.name] = st.slider(
                    param.label,
                    int(param.min_value),
                    int(param.max_value),
                    int(param.default),
                    int(param.step),
                    key=f"param_{param.name}"
                )
            elif param.param_type == "bool":
                param_values[param.name] = st.checkbox(
                    param.label,
                    value=param.default,
                    key=f"param_{param.name}"
                )
        
        st.markdown("---")
        st.markdown("### 📅 기간 설정")
        st.markdown("- **IS**: 2022년")
        st.markdown("- **OOS**: 2023년")
        st.markdown("- **FT**: 2024년")
        
        st.markdown("---")
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # 전략 초기화 (사용자 파라미터 적용)
    strategy = StrategyClass(**param_values)
    
    # 데이터 로드
    with st.spinner("데이터 로드 중..."):
        try:
            all_data = load_all_data()
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            st.stop()
    
    # 탭 생성 (전략 최적화 탭 추가 - Task 5)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 In-sample (2022)", "📈 Out-of-sample (2023)", "🎯 Forward Test (2024)", "🧪 전략 최적화 (Summary)"])
    
    phases = ['IS', 'OOS', 'FT']
    tabs = [tab1, tab2, tab3]
    
    # 전략 최적화 탭
    with tab4:
        st.header("📊 전략 파라미터 최적화 요약")
        st.info("Pre-calculated (Batch) 백테스트 결과를 비교합니다.")
        
        if st.button("🔄 저장된 결과 불러오기"):
            saved_results = res_mgr.list_combos(strategy_name)
            
            if saved_results:
                # DataFrame으로 변환
                summary_df = pd.DataFrame(saved_results)
                
                # 파라미터 컬럼 분리
                params_df = pd.json_normalize(summary_df['params'])
                display_df = pd.concat([params_df, pd.json_normalize(summary_df['metrics'])], axis=1)
                
                # 정렬 (평균 수익률 기준 내림차순)
                if 'avg_return' in display_df.columns:
                    display_df = display_df.sort_values('avg_return', ascending=False)
                
                # 포맷팅
                if 'win_rate' in display_df.columns:
                    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
                if 'avg_return' in display_df.columns:
                    display_df['avg_return'] = display_df['avg_return'].apply(lambda x: f"{x:+.2f}%")
                if 'total_return' in display_df.columns:
                    display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x*100:+.2f}%")
                
                st.write(f"총 {len(display_df)}개의 테스트 결과가 있습니다.")
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("저장된 일괄 백테스트 결과가 없습니다. generate_batch_results.py를 실행하세요.")
    

    
    for phase, tab in zip(phases, tabs):
        with tab:
            phase_data = all_data[phase]
            st.markdown(f'<div class="phase-header">{phase_data["name"]}</div>', unsafe_allow_html=True)
            
            # 종목 선택
            tickers = list(phase_data['data'].keys())
            if not tickers:
                st.warning("데이터가 없습니다.")
                continue
            
            selected_ticker = st.selectbox(
                "종목 선택",
                tickers,
                format_func=lambda x: f"{x.replace('.KS', '')} ({x})",
                key=f"ticker_{phase}"
            )
            
            # 배치 결과에서 일치하는 파라미터 찾기
            saved_combos = res_mgr.list_combos(strategy_name)
            matched_combo = None
            for combo in saved_combos:
                if combo['params'] == param_values:
                    matched_combo = combo
                    break
            
            if matched_combo:
                # 미리 계산된 결과 로드
                combo_id = matched_combo['id']
                combo_dir = res_mgr._get_combo_dir(strategy_name, combo_id)
                
                # 모든 종목 거래 로드 (CSV)
                all_trades_path = os.path.join(combo_dir, "모든종목", "trades.csv")
                if os.path.exists(all_trades_path):
                    all_trades_df = pd.read_csv(all_trades_path)
                    all_trades = all_trades_df.to_dict('records')
                    # 날짜 변환
                    for t in all_trades:
                        for date_key in ['date', '진입날짜', '청산날짜']:
                            if date_key in t and t[date_key]:
                                t[date_key] = pd.to_datetime(t[date_key])
                else:
                    all_trades = []
                
                metrics_dict = matched_combo['metrics']
                metrics = PerformanceMetrics(
                    avg_return_per_trade=metrics_dict['avg_return'],
                    win_rate=metrics_dict['win_rate'],
                    total_return=metrics_dict['total_return'],
                    total_trades=len(all_trades),
                    sharpe_ratio=0, cagr=0, mdd=0, expectancy=0 # 필요시 추가 계산
                )
            else:
                st.warning("일치하는 배치 결과가 없습니다. 실시간으로 계산합니다.")
                # 백테스트 실행 (실시간)
                metrics, all_trades = run_universe_backtest(
                    strategy,
                    phase_data['data'],
                    phase_data['start'],
                    phase_data['end'],
                    phase
                )
            
            # 선택된 종목 상세 분석 (배치 또는 실시간)
            trades = []
            if matched_combo:
                ticker_dir = os.path.join(res_mgr._get_combo_dir(strategy_name, combo_id), selected_ticker.replace(".KS", ""))
                trades_path = os.path.join(ticker_dir, "trades.csv")
                if os.path.exists(trades_path):
                    trades_df = pd.read_csv(trades_path)
                    trades = trades_df.to_dict('records')
                    # 날짜 변환
                    for t in trades:
                        for date_key in ['date', '진입날짜', '청산날짜']:
                            if date_key in t and t[date_key]:
                                t[date_key] = pd.to_datetime(t[date_key])
                        if 'return' in t: t['return'] = float(t['return'])
                        if '수익률' in t: t['수익률'] = float(t['수익률'])
                
                # result_df는 실시간 지표 계산을 위해 필요함 (보조지표 등)
                result_df = strategy.generate_signals(phase_data['data'][selected_ticker].copy())
            else:
                df = phase_data['data'][selected_ticker]
                result_df, trades = run_backtest_for_ticker(selected_ticker, df, strategy)
            
            # 선택된 종목 성능 지표 계산
            ticker_trades_for_metrics = [{
                'date': t['date'],
                'return': t['return'],
                'entry_price': t['entry_price'],
                'exit_price': t['exit_price']
            } for t in trades]
            
            # FT는 선택 종목에도 리스크 관리 적용
            if phase == 'FT' and ticker_trades_for_metrics:
                from backtest.risk_manager import RiskManager, RiskConfig
                rm = RiskManager(RiskConfig())
                adjusted_ticker_trades = []
                for t in ticker_trades_for_metrics:
                    adj_return = rm.calculate_adjusted_return(t['entry_price'], t['exit_price'])
                    adjusted_ticker_trades.append({
                        'date': t['date'],
                        'return': adj_return,
                        'entry_price': t['entry_price'],
                        'exit_price': t['exit_price']
                    })
                ticker_trades_for_metrics = adjusted_ticker_trades
            
            ticker_metrics = calculate_all_metrics(
                ticker_trades_for_metrics,
                phase_data['start'],
                phase_data['end'],
                10000000
            )
            
            # 전체 유니버스 성능 지표 표시
            st.markdown("#### 📊 전체 유니버스 성능")
            display_metrics(metrics, phase_data['name'])
            
            # 선택된 종목 성능 지표 표시
            st.markdown(f"#### 📈 선택 종목 ({selected_ticker.replace('.KS', '')}) 성능")
            display_metrics(ticker_metrics, f"{selected_ticker}")
            
            st.markdown("---")
            
            # 차트 표시
            col1, col2 = st.columns([7, 3])
            
            with col1:
                # 줄 상태 확인
                zoom_date = None
                if st.session_state.zoom_date is not None and st.session_state.zoom_phase == phase:
                    zoom_date = st.session_state.zoom_date
                
                # 고급 차트 생성 (Task 3, 4)
                # 주의: indicators는 result_df(백테스트 결과)에 포함되어 있음
                indicators = strategy.get_indicators(result_df)
                chart = create_advanced_chart(selected_ticker, result_df, trades, indicators)
                
                # 줌 적용 (Plotly zoom)
                if zoom_date:
                    from datetime import timedelta
                    zoom_start = zoom_date - timedelta(days=15)
                    zoom_end = zoom_date + timedelta(days=15)
                    chart.update_xaxes(range=[zoom_start, zoom_end])
                
                st.plotly_chart(chart, use_container_width=True, key=f"chart_{phase}")
                
                # 줄 초기화 버튼
                if zoom_date is not None:
                    if st.button("🔍 전체 차트 보기", key=f"reset_zoom_{phase}"):
                        st.session_state.zoom_date = None
                        st.session_state.zoom_phase = None
                        st.rerun()
            
            with col2:
                st.markdown("### 📋 거래 내역")
                
                if not trades:
                    st.info("해당 기간 거래 없음")
                else:
                    # 스크롤 가능한 컨테이너
                    with st.container(height=500):
                        for i, trade in enumerate(trades):
                            date_str = trade['date'].strftime('%Y-%m-%d') if hasattr(trade['date'], 'strftime') else str(trade['date'])
                            return_pct = trade['return'] * 100
                            color = "🟢" if return_pct > 0 else "🔴"
                            
                            col_btn, col_exp = st.columns([1, 4])
                            
                            with col_btn:
                                if st.button("🔍", key=f"zoom_{phase}_{i}", help="차트 확대"):
                                    st.session_state.zoom_date = trade['date']
                                    st.session_state.zoom_phase = phase
                                    st.rerun()
                            
                            with col_exp:
                                with st.expander(f"{color} {date_str} ({return_pct:+.1f}%)", expanded=False):
                                    st.markdown(trade['rationale'])
            
            # 전체 거래 통계 (Task 6)
            st.markdown("---")
            st.markdown("### 📊 전체 거래 통계")
            
            if trades:
                # DataFrame 변환
                trade_df = pd.DataFrame(trades)
                
                # 상단 요약 (Task 6-2)
                # 엔진에서 받은 metrics는 유니버스 전체, 여기는 선택된 ticker만.
                # ticker_metrics가 이미 계산되어 있음.
                avg_ret = ticker_metrics.avg_return_per_trade 
                win_rt = ticker_metrics.win_rate
                
                # 명시적 계산 (ticker_metrics가 없을 경우 대비)
                if 'return' in trade_df.columns:
                    avg_ret = trade_df['return'].mean() * 100
                    win_cnt = (trade_df['return'] > 0).sum()
                    win_rt = win_cnt / len(trade_df) * 100
                    
                st.markdown(f"#### 💡 평균 수익률: `{avg_ret:+.2f}%` | 승률: `{win_rt:.1f}%`")
                
                # 컬럼 매핑 (Task 6-1)
                column_map = {
                    'date': '진입날짜',
                    'exit_date': '청산날짜',
                    'ticker': '종목코드',
                    'type': '매매유형',
                    'entry_price': '진입가',
                    'exit_price': '청산가',
                    'return': '수익률',
                    'rationale': '매매근거'
                }
                
                # exit_date가 없을 수도 있으니 확인
                if 'exit_date' not in trade_df.columns:
                    trade_df['exit_date'] = None
                
                trade_df = trade_df.rename(columns=column_map)
                
                # 날짜 포맷팅
                for col in ['진입날짜', '청산날짜']:
                    if col in trade_df.columns:
                        trade_df[col] = pd.to_datetime(trade_df[col]).dt.strftime('%Y-%m-%d').fillna('-')
                
                # 숫자 포맷팅
                trade_df['수익률'] = trade_df['수익률'].apply(lambda x: f"{x*100:+.2f}%")
                trade_df['진입가'] = trade_df['진입가'].apply(lambda x: f"{x:,.0f}")
                trade_df['청산가'] = trade_df['청산가'].apply(lambda x: f"{x:,.0f}")
                
                # 주요 컬럼만 표시
                cols_to_show = ['진입날짜', '청산날짜', '진입가', '청산가', '수익률', '매매근거']
                st.dataframe(
                    trade_df[cols_to_show].sort_values('진입날짜', ascending=False), 
                    use_container_width=True, 
                    hide_index=True
                )
            else:
                st.info("거래 내역이 없습니다.")


if __name__ == "__main__":
    main()
