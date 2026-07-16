# NYC Yellow Taxi End-to-End Data Analysis

NYC Yellow Taxi Parquet 데이터를 이용해 Pandas·Polars 비교, EDA, 시각화,
통계 검정, 머신러닝 Pipeline 학습과 보고서 생성을 자동화한 프로젝트입니다.

## 주요 기능

- Pandas와 Polars 데이터 로딩 결과 비교
- 결측치·중복·비정상 거리 및 요금 처리
- 평균·표준편차·분위수와 변수 간 상관계수 출력
- Seaborn 정적 차트와 Plotly 인터랙티브 HTML 생성
- 장거리·단거리 그룹 요금의 Welch 독립표본 t-test
- 장거리 운행 분류 Pipeline 학습 및 Accuracy·F1 출력
- `joblib` 모델과 `report.md` 자동 저장

## 실행 환경

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NYC TLC의 `yellow_tripdata_2026-05.parquet` 파일을 프로젝트 루트에 준비합니다.
원본 데이터는 저장소 용량과 재배포를 고려해 Git에 포함하지 않습니다.

## 실행

```bash
python 울산_1반_김윤성.py yellow_tripdata_2026-05.parquet
```

빠른 검증을 위해 학습 표본 수를 줄일 수도 있습니다.

```bash
python 울산_1반_김윤성.py yellow_tripdata_2026-05.parquet --max-training-rows 200000
```

실행 결과는 `project_outputs/`에 저장됩니다.

- `report.md`: 자동 생성 분석 보고서
- `metrics.json`: 데이터 품질·통계·모델 지표
- `seaborn_fare_distribution.png`: 정적 차트
- `plotly_hourly_payment.html`: 인터랙티브 차트
- `taxi_long_trip_pipeline.joblib`: 학습된 Pipeline 모델

## 현재 시각 대기 지역 Top 3 추천

`recommend_waiting_zones.py`는 운행 전에 알 수 있는 요일·시간·승차 지역과
과거 시간대별 수요만 사용해 전체 승차 및 장거리 승차 건수를 예측합니다.
기본 장거리 기준은 5 miles이며 결과창은 HTML로 생성됩니다.

운영 모델은 5월 1~25일 학습, 26~31일 테스트의 단일 시간 분할을 사용합니다.
5-fold의 개선 폭이 비용 대비 작았기 때문입니다. 연구용 비교가 필요할 때만
`--compare-5fold` 옵션으로 두 방식을 다시 계산할 수 있습니다.

```bash
python recommend_waiting_zones.py yellow_tripdata_2026-05.parquet
open recommendation_outputs/current_recommendations.html
```

원하는 뉴욕 현지 시각을 지정해 모의실험할 수도 있습니다.

```bash
python recommend_waiting_zones.py yellow_tripdata_2026-05.parquet --at "2026-07-17 22:00"
```

추천 산출물:

- `current_recommendations.html`: Top 3 추천 카드와 상위 10개 지역 차트
- `current_zone_ranking.csv`: 전체 지역 예측 순위
- `waiting_zone_demand_models.joblib`: 전체·장거리 수요 Pipeline 모델
- `recommendation_metrics.json`: MAE·RMSE·Precision@3

기본 운영 모델의 최종 테스트 결과:

- 장거리 수요 MAE: 1.4118건
- 장거리 수요 RMSE: 3.8762건
- 추천 Precision@3: 75.00%

## 검증 결과

- 전체 데이터: 4,090,836행 × 20열
- 정제 후 데이터: 3,980,145행
- Accuracy: 0.9318
- F1 score: 0.9320
