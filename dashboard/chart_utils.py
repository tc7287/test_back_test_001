
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

def create_advanced_chart(
    ticker: str, 
    df: pd.DataFrame, 
    trades: List[Dict], 
    indicators: List[Dict[str, Any]]
) -> go.Figure:
    """
    고급 차트 생성 (캔들스틱 + 거래량 + 보조지표 + 매매구간)
    
    Tasks:
    3. 보조 지표 스택 (Overlay & Secondary Panels)
    4. 매도 시그널 및 수익/손실 구간 표시
    """
    
    # 1. 패널 구성 분석
    # Secondary 지표들을 그룹핑하여 필요한 하단 패널 수 계산
    secondary_panels = []
    secondary_groups = {} # {panel_name: [indicators]}
    
    for ind in indicators:
        if ind.get('type') == 'secondary':
            name = ind['name']
            # 그룹핑 로직: 이름에 포함된 키워드로 묶기 (간단한 버전)
            # RSI 관련은 RSI 패널로
            group_name = name
            if 'RSI' in name or 'Over' in name:
                group_name = 'RSI'
            
            # 다른 지표가 있다면 여기서 추가 로직 필요 (예: MACD)
            
            if group_name not in secondary_groups:
                secondary_groups[group_name] = []
                secondary_panels.append(group_name)
            secondary_groups[group_name].append(ind)
            
    # 행 구성: [Price(0.6), Volume(0.2), Ind1(0.2), Ind2(0.2)...]
    # 기본 2개 행(가격, 거래량) + 보조지표 패널 수
    n_secondary = len(secondary_panels)
    row_heights = [0.6, 0.15] + [0.25] * n_secondary
    
    # 높이 정규화
    total_h = sum(row_heights)
    row_heights = [h/total_h for h in row_heights]
    
    subplot_titles = [f'{ticker} 주가', '거래량'] + secondary_panels
    
    fig = make_subplots(
        rows=2 + n_secondary,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # 2. 메인 차트 (Row 1) - 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 3. 거래량 (Row 2) - Bar
    colors = ['red' if row['open'] - row['close'] >= 0 else 'green' for i, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. 보조 지표 표시
    # 4.1 Overlay 지표 (Row 1)
    for ind in indicators:
        if ind.get('type') == 'overlay':
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ind['data'],
                    name=ind['name'],
                    line=dict(color=ind.get('color', 'blue'), width=1),
                    mode='lines'
                ),
                row=1, col=1
            )
            
    # 4.2 Secondary 지표 (Row 3+)
    for i, group_name in enumerate(secondary_panels):
        row_idx = 3 + i
        group_inds = secondary_groups[group_name]
        
        for ind in group_inds:
            line_style = dict(color=ind.get('color', 'blue'), width=1)
            if ind.get('dash'):
                line_style['dash'] = ind['dash']
                
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ind['data'],
                    name=ind['name'],
                    line=line_style,
                    mode='lines'
                ),
                row=row_idx, col=1
            )
            
            # y축 범위 설정 (예: RSI 0~100)
            if 'axis_range' in ind:
                fig.update_yaxes(range=ind['axis_range'], row=row_idx, col=1)

    # 5. 매매 시각화 (Task 4)
    # 5.1 시그널 마커 (매수: 삼각형 위, 매도: 삼각형 아래) 및 구간 표시
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    
    for t in buy_trades:
        entry_date = t['date']
        exit_date = t.get('exit_date')
        
        # 수익 여부에 따른 색상 (투명도 적용)
        is_profit = t['return'] > 0
        fill_color = 'rgba(255, 0, 0, 0.1)' if is_profit else 'rgba(0, 0, 255, 0.1)'
        
        # 구간 표시 (add_vrect)
        if exit_date and pd.notna(exit_date):
            fig.add_vrect(
                x0=entry_date, x1=exit_date,
                fillcolor=fill_color, opacity=1, layer="below", line_width=0,
            )
            
            # 매도 마커
            fig.add_trace(
                go.Scatter(
                    x=[exit_date], y=[t.get('exit_price')],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='blue'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
    if buy_trades:
        # 매수 마커 (한 번에 추가)
        fig.add_trace(
            go.Scatter(
                x=[t['date'] for t in buy_trades],
                y=[t['entry_price'] for t in buy_trades],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color='red'),
                showlegend=False
            ),
            row=1, col=1
        )

    # ... (일단 차트 코드 마무리 하고 전략 수정하러 가야 함)
    
    fig.update_layout(height=600 + (n_secondary * 200), xaxis_rangeslider_visible=False)
    
    return fig
