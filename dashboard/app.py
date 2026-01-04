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


def run_backtest_for_ticker(ticker: str, df: pd.DataFrame, strategy):
    """단일 종목 백테스트 실행 및 상세 정보 반환 (모든 전략 호환)"""
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
                'return': row['returns'],
                # 전략의 get_trade_rationale 메서드 사용
                'rationale': strategy.get_trade_rationale(row_series, ticker)
            }
            trades.append(trade)
    
    return result_df, trades


def create_chart_with_signals(df: pd.DataFrame, trades: list, ticker: str, zoom_date=None):
    """봘수/매도 시그널이 표시된 차트 생성
    
    Args:
        zoom_date: 줄 중심 날짜 (상하 15일 범위 = 약 1개월)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} 주가 차트', '거래량')
    )
    
    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )
    
    # 목표가 라인
    if 'target_price' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['target_price'],
                mode='lines',
                name='목표가',
                line=dict(color='#fbbf24', width=1, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # 매수 시그널 마커
    buy_dates = [t['date'] for t in trades]
    buy_prices = [t['entry_price'] for t in trades]
    buy_returns = [t['return'] for t in trades]
    buy_texts = [f"매수: ₩{t['entry_price']:,.0f}<br>수익률: {t['return']*100:+.2f}%<br>클릭하여 확대" for t in trades]
    
    # 수익/손실에 따른 색상
    marker_colors = ['#22c55e' if r > 0 else '#ef4444' for r in buy_returns]
    
    fig.add_trace(
        go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='매수 시점',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color=marker_colors,
                line=dict(width=2, color='white')
            ),
            text=buy_texts,
            hovertemplate='%{text}<extra></extra>',
            customdata=list(range(len(trades)))
        ),
        row=1, col=1
    )
    
    # 거래량
    colors = ['#22c55e' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef4444' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='거래량',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 레이아웃
    layout_config = dict(
        height=600,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # 줄 설정 (약 1개월 = 30일)
    if zoom_date is not None:
        from datetime import timedelta
        zoom_start = zoom_date - timedelta(days=15)
        zoom_end = zoom_date + timedelta(days=15)
        layout_config['xaxis'] = dict(range=[zoom_start, zoom_end])
        layout_config['xaxis2'] = dict(range=[zoom_start, zoom_end])
    
    fig.update_layout(**layout_config)
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
    
    return fig


def display_metrics(metrics: PerformanceMetrics, phase_name: str):
    """성능 지표 표시"""
    cols = st.columns(4)
    
    with cols[0]:
        delta_color = "normal" if metrics.total_return >= 0 else "inverse"
        st.metric("총 수익률", f"{metrics.total_return:+.1f}%", 
                  delta=f"CAGR {metrics.cagr:+.1f}%", delta_color=delta_color)
    
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
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 In-sample (2022)", "📈 Out-of-sample (2023)", "🎯 Forward Test (2024)"])
    
    phases = ['IS', 'OOS', 'FT']
    tabs = [tab1, tab2, tab3]
    
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
            
            # 백테스트 실행
            df = phase_data['data'][selected_ticker]
            result_df, trades = run_backtest_for_ticker(selected_ticker, df, strategy)
            
            # 전체 성능 지표 계산
            all_trades = []
            for ticker, ticker_df in phase_data['data'].items():
                _, ticker_trades = run_backtest_for_ticker(ticker, ticker_df, strategy)
                for t in ticker_trades:
                    all_trades.append({
                        'date': t['date'], 
                        'return': t['return'],
                        'entry_price': t['entry_price'],
                        'exit_price': t['exit_price']
                    })
            
            # FT는 리스크 관리 적용 (슬리피지/수수료 반영)
            if phase == 'FT' and all_trades:
                from backtest.risk_manager import RiskManager, RiskConfig
                rm = RiskManager(RiskConfig())
                adjusted_trades = []
                for t in all_trades:
                    adj_return = rm.calculate_adjusted_return(t['entry_price'], t['exit_price'])
                    adjusted_trades.append({
                        'date': t['date'],
                        'return': adj_return,
                        'entry_price': t['entry_price'],
                        'exit_price': t['exit_price']
                    })
                all_trades = adjusted_trades
            
            metrics = calculate_all_metrics(
                all_trades,
                phase_data['start'],
                phase_data['end'],
                10000000
            )
            
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
                
                chart = create_chart_with_signals(result_df, trades, selected_ticker, zoom_date)
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
            
            # 전체 거래 통계
            st.markdown("---")
            st.markdown("### 📊 전체 거래 통계")
            
            if trades:
                trade_df = pd.DataFrame([{
                    '날짜': t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else t['date'],
                    '진입가': f"₩{t['entry_price']:,.0f}",
                    '청산가': f"₩{t['exit_price']:,.0f}",
                    '수익률': f"{t['return']*100:+.2f}%",
                    '결과': '✅ 수익' if t['return'] > 0 else '❌ 손실'
                } for t in trades])
                
                st.dataframe(trade_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
