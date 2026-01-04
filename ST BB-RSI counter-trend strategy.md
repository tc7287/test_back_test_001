# 📈 주식 자동 매매 전략 보고서: BB-RSI 역추세 전략

본 전략은 가격의 변동성을 측정하는 **불린저 밴드(Bollinger Bands)**와 지표의 과매수/과매도를 나타내는 **RSI(상대강도지수)**를 결합하여 승률을 높인 역추세 매매 기법입니다.

---

## 1. 전략 개요 (Strategy Concept)
- **성격**: 평균 회귀 (Mean Reversion)
- **핵심 원리**: 주가가 통계적 변동 범위(BB 하단)를 벗어나고, 심리적 저점(RSI 과매도)에 도달했을 때의 반등을 노림.
- **적정 주기**: 15분봉 또는 1시간봉 (데이 트레이딩 및 스윙에 최적)

---

## 2. 설정 파라미터 (Parameters)

| 구분 | 항목 | 설정값 | 비고 |
| :--- | :--- | :--- | :--- |
| **Bollinger Bands** | 기간 (Period) | **20** | 이동평균선 기준 |
| | 표준편차 (StdDev) | **2** | 가격의 95.4% 포함 범위 |
| **RSI** | 기간 (Period) | **14** | 기본 설정값 |
| | 과매수 기준 | **70** | 하락 반전 가능성 |
| | 과매도 기준 | **30** | 상승 반전 가능성 |

---

## 3. 매매 로직 (Trading Logic)

### ✅ 매수 진입 (Buy Entry)
다음 두 조건이 **동시에 만족(AND)**될 때 매수 주문을 실행합니다.
1. **Price Condition**: 현재가(Close) ≤ 불린저 밴드 하단선 (Lower Band)
2. **Indicator Condition**: RSI(14) ≤ 30

### ✅ 매수 청산 (Exit Strategy)
리스크 관리와 수익 극대화를 위해 분할 또는 전량 매도를 실행합니다.
- **익절 (Take Profit)**
  - 1차: 가격이 불린저 밴드 **중심선**에 도달 시 (보수적)
  - 2차: 가격이 불린저 밴드 **상단선**에 도달 시 (공격적)
- **손절 (Stop Loss)**
  - 진입가 대비 **-2% ~ -3%** 이탈 시 즉시 매도 (필수)

---

## 4. 알고리즘 구현 참고 (Python 예시 수식)

```python
# 불린저 밴드 계산
upper_band = sma20 + (std_dev * 2)
lower_band = sma20 - (std_dev * 2)

# 매수 조건 예시
if (current_price <= lower_band) and (current_rsi <= 30):
    execute_buy_order()