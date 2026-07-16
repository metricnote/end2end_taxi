# NYC Yellow Taxi End-to-End 분석 보고서

## 1. 프로젝트 개요

- 데이터: `yellow_tripdata_2026-05.parquet`
- 원본 크기: 4,090,836행 × 20열
- 목표: Pandas·Polars 비교부터 시각화, 통계 검정, ML Pipeline 저장까지 자동화
- 분류 목표: `trip_distance`가 정제 데이터 중앙값 1.900 miles 이상인 장거리 운행

## 2. 데이터 준비 및 EDA

- Pandas/Polars 행·열 수 일치: True
- Pandas/Polars 평균 총액: 30.485 / 30.485
- 중복 행: Pandas 0, Polars 0
- 주요 결측치: passenger_count: 955,371, RatecodeID: 955,371, store_and_fwd_flag: 955,371, congestion_surcharge: 955,371, Airport_fee: 955,371
- 비정상 거리·요금·소요시간 제거 후: 3,980,145행
- 거리와 기본요금의 상관계수: 0.8551

## 3. 시각화

- Seaborn 정적 차트: `seaborn_fare_distribution.png`
- Plotly 인터랙티브 차트: `plotly_hourly_payment.html`
- 두 차트 모두 제목과 축 레이블을 포함한다.

## 4. 통계 검정

- 검정: 장거리 vs 단거리 운행의 `fare_amount` Welch 독립표본 t-test
- 단거리 평균 요금: $12.056
- 장거리 평균 요금: $30.653
- t-statistic: -1367.746041
- p-value: 0
- 해석: p-value < 0.05이므로 두 그룹의 평균 요금 차이는 통계적으로 유의하다.

## 5. ML Pipeline

- Pipeline: 수치 결측치 대체·표준화 + 범주 결측치 대체·인코딩 + HistGradientBoostingClassifier
- 평가 표본: 500,000행
- 정확도(Accuracy): 0.9333
- F1 score: 0.9336
- 저장 모델: `taxi_long_trip_pipeline.joblib`

## 6. 결론

Pandas와 Polars의 핵심 결과가 일치했으며, 시각화와 t-test를 통해 거리 그룹별 요금 차이를 확인했다.
저장된 Pipeline은 동일한 전처리와 모델을 함께 재사용할 수 있다.
