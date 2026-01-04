 자동매매 시스템 설계서 / 제안서 (Python 기반)
📌 문서 개요

본 문서는 Python 기반으로 구현할 전략/백테스트/실매매 자동매매 시스템의 전체 아키텍처, 모듈 설계, 흐름, 개발 시 유의 사항, 그리고 오프라인(백테스트)/온라인(실매매) 실행 시나리오를 정의한다.

핵심 목표는 다음과 같다:

동일한 전략 로직을 **오프라인(백테스트)**과 온라인(실매매) 모두에서 재사용

RiskManager를 중심으로 일관적 리스크 관리

전략 신호(매수/매도/손절/익절)의 구조화

계좌 기반 리스크 관리(1% per trade, 5% total exposure 등)

확장 가능한 전략 Pool, 브로커 Adapter, 데이터 Provider 구조

# 1. 시스템 전체 구조
📌 전체 아키텍처 다이어그램
               +------------------------+
               |      Controller        |
               |  Offline / Online      |
               +-----------+------------+
                           |
         +-----------------+-------------------+
         |                                     |
+--------v--------+                    +-------v--------+
|   Strategy Pool |                    |   Backtester   |  ← Offline 전용
+--------+--------+                    +-------+--------+
         |                                     |
         | uses                                | uses
         v                                     v
 +-------+----------+                 +---------+---------+
 |    Data Module  |                 |  Risk Manager      |
 +-------+----------+                 +---------+---------+
         |                                     |
         |                                     |
         |                               +-----v------+
         |                               |  Broker    |
         v                               | (Adapter)  |
+--------+----------+                    +------------+
| Monitoring/Report |
+-------------------+

# 2. 모듈 상세 설계
## 2.1 Data Module (데이터 모듈)
✔ 역할

OHLCV 가격 데이터 제공

시총 정보, 종목 리스트 제공

캐싱 및 통합 인터페이스 제공

✔ 주요 기능
class DataProvider:
    def get_ohlcv(self, symbol, start, end, freq="1D") -> pd.DataFrame:
        ...

    def get_latest_bar(self, symbol) -> pd.Series:
        ...

    def get_universe(self, date) -> list[str]:
        ...  # 예: KOSPI 시총 상위 20

✔ 개발 유의사항

모든 전략에서 동일한 OHLCV 스펙 사용 (index=datetime, open/high/low/close/volume)

데이터 결측치 처리 일관화

캐싱 활용하여 성능 최적화

## 2.2 Strategy Module (전략 모듈)
✔ 역할

매수/매도/보유(HOLD) **시그널 생성**

손절선 / 익절선 / 트레일링 스탑 등 **전략적 규칙 정의 및 업데이트**

✔ 베이스 클래스
class Signal:
    symbol: str
    action: str      # "BUY", "SELL", "HOLD"
    entry_price: float
    stop_price: float
    take_profit: float
    meta: dict

class BaseStrategy(ABC):
    def generate_signals(self, date, universe, price_history, current_positions):
        raise NotImplementedError

✔ 전략의 책임

- 신규 진입 시점 결정 (예: RSI < 20 → BUY)
- 청산 시점 결정 (손절/익절 기준에 도달 시 SELL 시그널 생성)
- 손절선 결정 (예: 10%)
- 익절선 결정 (예: 20%)
- 트레일링 손절/익절 조정 (예: BEP 이상으로 올리기, TP 상향 조정)
- 신호 강도 계산 (RSI 기반 등)

✔ 개발 유의사항

- 전략은 **“무엇을(심볼), 언제(BUY/SELL/HOLD), 어디까지(stop/tp)”**만 책임진다.
- 전략은 계좌 전체 리스크(잔고, 현재 포지션 수, 총 익스포저)를 모른다  
  → **리스크 관리 기능은 RiskManager에게 위임**
- 전략은 반드시 `stop_price`, `take_profit`을 제공해야 한다 (포지션 사이징 및 리스크 계산에 필요)

## 2.3 Risk Manager (리스크 매니저)
✔ 역할

전략이 생성한 BUY/SELL/HOLD 시그널을 입력으로 받아:

- 포지션 사이징 (R = 1% per trade 등)
- 계좌 전체 리스크 제한 적용 (예: 5%)
- 최대 포지션 수 제한 (예: 5종목)
- 신호 과잉일 때 종목 선택 (강도 상위 N개만 채택)
- (옵션) 손절/익절 기준을 활용해 자동 청산 정책 추가 가능

✔ 주요 기능
class RiskManager:
    def size_position(self, entry_price, stop_price, nav, risk_per_trade=0.01):
        risk_amount = nav * risk_per_trade
        risk_per_share = entry_price - stop_price
        qty = floor(risk_amount / risk_per_share)
        return max(qty, 0)

    def filter_signals(self, signals, current_positions, nav):
        ...

✔ 개발 유의사항

전략이 준 `stop_price` / `take_profit` 기반으로 포지션 리스크 계산

전략이 생성한 BUY/SELL 시그널을 **그대로 신뢰하되**,  
리스크 한도(거래당, 계좌 전체, 최대 포지션 수)를 넘지 않도록 **실행 여부와 수량만 결정**

트레일링 스탑은 전략에서 stop/tp를 업데이트하고,  
RiskManager는 최신 stop/tp 값을 사용해 리스크를 재계산

실제 시장에서는 Gap Risk 존재 → 추후 Gap Factor 도입 가능

## 2.4 Backtest Module (오프라인 백테스트)
### 2.4.1 Portfolio Backtest (포트폴리오 기반 백테스트)
✔ 역할

- 전략 + RiskManager + 데이터 기반으로 **실전 구조에 가까운 과거 시뮬레이션**
- 여러 종목을 동시에 보유하면서, 계좌 단위 리스크 / 최대 포지션 수 등을 재현
- 포지션, NAV, 연도별 수익률 기록

✔ 주요 흐름
for date in trading_days:
    data = data_provider.get_ohlcv(...)
    signals = strategy.generate_signals(...)
    sized_orders = risk_manager.filter_signals(signals, current_positions, nav)
    executed = simulate_fills(sized_orders)
    update_portfolio(executed)

✔ 개발 유의사항

- 체결 로직 일관화 (대부분 D일 시가 또는 다음날 시가)
- 슬리피지/수수료 반영
- 종가 기준 손절/익절 체크 우선순위 명확히

### 2.4.2 Single-Symbol Backtest (단일 종목 전략 검증용)
✔ 역할

- **포트폴리오/리스크 제약을 제거**한 상태에서,
  개별 전략 로직이 단일 종목 OHLCV에 대해 얼마나 잘 작동하는지 평가
- 전략 자체의 품질(진입/청산 로직, stop/tp, 트레일링 등)을 독립적으로 검증

✔ 주요 개념

- 입력:
  - symbol: 단일 종목 코드
  - ohlcv: 해당 종목의 OHLCV DataFrame (index=datetime, [open, high, low, close, volume])
  - strategy: BaseStrategy 구현체 (예: RSIMeanReversionStrategy)
  - initial_cash, risk_per_trade: 단일 종목 테스트용 자본/리스크 설정
- 출력:
  - equity_curve: 날짜별 자산 곡선
  - trades: 각 트레이드의 진입/청산/수익률 리스트

✔ 예시 클래스 (요약)

SingleSymbolBacktester:
    def __init__(self, strategy, initial_cash=10_000_000): ...

    def run(self, symbol, ohlcv, risk_per_trade=0.01) -> SingleSymbolResult:
        """
        - 단일 종목 OHLCV만 보고 전략 시그널(BUY/SELL)을 따라가는 간단한 백테스트
        - 포트폴리오 제약 없이, 전략 로직만 반영
        """

## 2.5 Broker Module (브로커/체결 모듈)
✔ 역할

실매매 Broker Adapter

모의매매(Paper Broker) 구현

주문/체결 조회

✔ 인터페이스
class BrokerInterface:
    def place_order(self, order) -> str: ...
    def get_positions(self) -> list[Position]: ...
    def get_cash(self) -> float: ...

## 2.6 Controller Module
✔ 역할

오프라인/온라인 실행 흐름 조정

백테스트 실행 or 실시간 자동매매 실행

# 3. 오프라인 / 온라인 시나리오
## 3.1 Offline Phase (백테스트)
✔ 목적

전략 자체의 성능 검증

실전 구조 포함한 Execution Simulation 수행

✔ 실행 흐름

데이터 로딩

전략 신호 생성

RiskManager 수량 산정 (리스크 1%, 계좌 5%)

체결 시뮬레이션

손절/익절 체크 및 자동 청산

포트/NAV 업데이트

성과 기록 (연별/월별 수익률, MDD 등)

📌 예시 시나리오 (RSI 전략)

RSI < 20 → 매수

손절 10%, 익절 20%

일정 상승 후 손절선을 0%로 올림

포지션 최대 5종목

거래당 리스크 = 1%

백테스터는 RiskManager를 이용해:

적절한 수량 계산

과도 신호는 상위 RSI 강도 기준 상위 5개만 채택

손절선이 올라가면 리스크 재계산

## 3.2 Online Phase (실매매)
✔ 목적

검증된 전략을 실전 환경에서 자동 실행

실시간 리스크 제한 준수

안전한 주문 처리

✔ 온라인 실행 흐름

전략 스케줄 실행 (예: 09:05)

가격 데이터 최신화

전략 신호 생성

RiskManager로 주문 수량 계산

계좌 리스크 초과 시 매수 제한

Broker API로 주문 실행

손절/익절 조건 충족 시 자동 청산

로그 및 Slack/Telegram 알림

# 4. 전략 개발/튜닝을 위한 백테스트 운용 방법
📌 백테스트 3단계 구조 (정석)

In-sample (전략 개발 구간)

전체 데이터의 60~70%

전략 설계/파라미터 튜닝

상승/하락/횡보 포함

Out-of-sample (검증 구간)

20~30%

과적합 여부 확인

Forward Test (최종 모의고사)

최근 1~2년

실전 적합성 검증

# 5. 개발 시 유의 사항
✔ 1) 전략과 RiskManager의 책임 분리

전략은 “언제 사고/팔지”, “손절·익절선”만 제공

RiskManager는 수량·계좌 리스크·종목 선택을 담당

이 분리가 되어야 유지보수가 쉬움

✔ 2) 백테스트와 실매매는 동일한 RiskManager 로직을 사용

실전과 백테스트가 달라지면 성능 괴리 발생

단, 체결 슬리피지는 실전에서 더 크게 적용 가능

✔ 3) 신호 강도 기반 필터링 로직 구현

예: RSI 전략 → RSI 낮을수록 신호 강함

매수 신호가 7개면 상위 5개만 선택

✔ 4) 트레일링 손절선/익절선은 전략에서 업데이트

RiskManager는 단순히 반영만 해야 함.

✔ 5) 포지션 리스크 계산은 매일 재평가

손절선이 올라가면 해당 포지션 리스크 = 0으로 감소
→ 계좌 리스크 여유 증가 → 신규 매수 허용

# 6. Example: RSI Mean Reversion Strategy 설계
✔ 매수 조건

RSI(14) < 20

코스피 시총 상위 20개 종목 대상

✔ 초기 설정

손절: -10%

익절: +20%

✔ 트레일링 로직

수익률 +10% 이상 시 손절선을 Break-even으로 올림

수익률 +20% 도달 시 익절 목표를 +30%로 조정

✔ 리스크 관리

거래당 리스크 = 1%

계좌 전체 리스크 = 5%

최대 포지션 = 5종목

# 7. 파일 구조 예시
/project
 ├── data/
 ├── core/
 │    ├── strategy/
 │    ├── risk/
 │    ├── backtest/
 │    ├── broker/
 │    ├── controller/
 │    └── common/
 ├── config/
 ├── notebooks/
 ├── reports/
 └── main.py

# 8. 마무리

본 설계서는 전략 개발–백테스트–실매매 자동화 전체 파이프라인을
Python 기반으로 구축하기 위한 기반 문서이며,

모듈 간 역할 분리

오프라인/온라인 로직 분리

실전 리스크 관리 반영

재사용 가능한 구조