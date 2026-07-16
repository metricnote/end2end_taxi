"""SKALA 4기 울산 1반 김윤성 - End2End 데이터 분석 프로젝트.

NYC Yellow Taxi Parquet 데이터를 Pandas와 Polars로 비교 분석하고,
EDA·시각화·통계 검정·분류 Pipeline·모델 저장·Markdown 보고서 생성을 자동화한다.

변경내역:
Pandas/Polars 데이터 품질 비교와 기본 EDA 구현
Seaborn·Plotly 시각화, t-test, ML Pipeline 구현
joblib 모델 및 report.md 자동 저장 기능 추가
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_ind
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
except ModuleNotFoundError as error:
    raise SystemExit(
        f"필수 라이브러리가 없습니다: {error.name}\n"
        "설치: python -m pip install pandas polars pyarrow numpy matplotlib seaborn "
        "plotly scipy scikit-learn joblib"
    ) from error


REQUIRED_COLUMNS = {
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "RatecodeID", "PULocationID",
    "DOLocationID", "payment_type", "fare_amount", "tip_amount", "total_amount",
}
NUMERIC_FEATURES = ["fare_amount", "passenger_count", "duration_minutes", "pickup_hour"]
CATEGORICAL_FEATURES = [
    "VendorID", "RatecodeID", "PULocationID", "DOLocationID",
    "payment_type", "pickup_dayofweek",
]
TARGET = "is_long_trip"


def validate_columns(columns: list[str]) -> None:
    """분석에 필요한 컬럼이 빠졌으면 명확한 오류를 발생시킨다."""
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")


def load_compare_clean(parquet_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pandas와 Polars로 같은 Parquet를 읽어 품질·기술통계를 비교하고 정제한다."""
    polars_frame = pl.read_parquet(parquet_path)
    validate_columns(polars_frame.columns)
    polars_duplicates = int(polars_frame.is_duplicated().sum())
    polars_missing_row = polars_frame.null_count().row(0, named=True)
    polars_missing = {key: int(value) for key, value in polars_missing_row.items() if value}
    polars_total_mean = float(polars_frame["total_amount"].mean())
    polars_shape = [polars_frame.height, polars_frame.width]
    del polars_frame

    pandas_frame = pd.read_parquet(parquet_path)
    validate_columns(pandas_frame.columns.tolist())
    pandas_duplicates = int(pandas_frame.duplicated().sum())
    pandas_missing = {key: int(value) for key, value in pandas_frame.isna().sum().items() if value}
    pandas_total_mean = float(pandas_frame["total_amount"].mean())
    if not np.isclose(pandas_total_mean, polars_total_mean, rtol=1e-12):
        raise ValueError("Pandas와 Polars의 total_amount 평균이 일치하지 않습니다.")

    # 물리적으로 불가능한 시간·거리·요금과 완전 중복 행을 분석 대상에서 제외한다.
    frame = pandas_frame.drop_duplicates().copy()
    frame["duration_minutes"] = (
        frame["tpep_dropoff_datetime"] - frame["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    frame["pickup_hour"] = frame["tpep_pickup_datetime"].dt.hour
    frame["pickup_dayofweek"] = frame["tpep_pickup_datetime"].dt.dayofweek
    valid = (
        frame["trip_distance"].between(0, frame["trip_distance"].quantile(0.999))
        & frame["fare_amount"].ge(0)
        & frame["total_amount"].ge(0)
        & frame["duration_minutes"].between(1, 240)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("정제 후 분석할 행이 없습니다.")

    metrics: dict[str, object] = {
        "pandas_shape": list(pandas_frame.shape),
        "polars_shape": polars_shape,
        "pandas_duplicates": pandas_duplicates,
        "polars_duplicates": int(polars_duplicates),
        "pandas_missing": pandas_missing,
        "polars_missing": polars_missing,
        "pandas_total_mean": pandas_total_mean,
        "polars_total_mean": polars_total_mean,
        "clean_rows": len(frame),
    }
    del pandas_frame
    return frame, metrics


def descriptive_analysis(frame: pd.DataFrame) -> dict[str, object]:
    """평균·표준편차·분위수와 수치형 상관계수를 계산하고 출력한다."""
    numeric = ["trip_distance", "fare_amount", "tip_amount", "total_amount", "duration_minutes"]
    description = frame[numeric].describe(percentiles=[0.25, 0.5, 0.75]).round(3)
    correlation = frame[numeric].corr().round(4)
    print("\n[기술통계: 평균·표준편차·분위수]")
    print(description.to_string())
    print("\n[상관계수]")
    print(correlation.to_string())
    return {
        "description": description.to_dict(),
        "correlation": correlation.to_dict(),
    }


def create_seaborn_chart(frame: pd.DataFrame, output_path: Path) -> None:
    """장거리 여부에 따른 요금 분포를 비교하는 Seaborn 정적 차트를 저장한다."""
    sns.set_theme(
        style="whitegrid",
        rc={"font.family": "AppleGothic", "axes.unicode_minus": False},
    )
    plot_frame = frame.sample(min(120_000, len(frame)), random_state=42).copy()
    upper = plot_frame["fare_amount"].quantile(0.99)
    plot_frame = plot_frame[plot_frame["fare_amount"].between(0, upper)]
    plot_frame["trip_group"] = np.where(plot_frame[TARGET].eq(1), "Long trip", "Short trip")

    fig, axis = plt.subplots(figsize=(11, 6))
    sns.histplot(
        data=plot_frame, x="fare_amount", hue="trip_group", bins=45,
        stat="density", common_norm=False, element="step", kde=True, ax=axis,
    )
    axis.set_title("Fare Distribution by Trip-distance Group")
    axis.set_xlabel("Fare amount (USD, lower 99%)")
    axis.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_plotly_chart(frame: pd.DataFrame, output_path: Path) -> None:
    """시간대·결제유형별 평균 총액을 Plotly 인터랙티브 HTML로 저장한다."""
    hourly = (
        frame.groupby(["pickup_hour", "payment_type"], as_index=False)
        .agg(average_total=("total_amount", "mean"), trips=("total_amount", "size"))
    )
    hourly["payment_type"] = hourly["payment_type"].astype(str)
    chart = px.line(
        hourly, x="pickup_hour", y="average_total", color="payment_type",
        markers=True, hover_data=["trips"],
        title="Average Total Amount by Pickup Hour and Payment Type",
        labels={
            "pickup_hour": "Pickup hour", "average_total": "Average total amount (USD)",
            "payment_type": "Payment type", "trips": "Trip count",
        },
    )
    chart.update_layout(hovermode="x unified")
    chart.write_html(output_path, include_plotlyjs=True)


def run_t_test(frame: pd.DataFrame) -> dict[str, float | str]:
    """장거리·단거리 그룹의 평균 요금 차이를 Welch t-test로 검정하고 해석한다."""
    short_fare = frame.loc[frame[TARGET].eq(0), "fare_amount"].dropna()
    long_fare = frame.loc[frame[TARGET].eq(1), "fare_amount"].dropna()
    statistic, p_value = ttest_ind(short_fare, long_fare, equal_var=False)
    interpretation = (
        "p-value < 0.05이므로 두 그룹의 평균 요금 차이는 통계적으로 유의하다."
        if p_value < 0.05
        else "p-value >= 0.05이므로 두 그룹의 평균 요금 차이는 통계적으로 유의하지 않다."
    )
    print("\n[Welch t-test: 단거리 vs 장거리 fare_amount]")
    print(f"t-statistic={statistic:.6f}, p-value={p_value:.6g}")
    print("해석:", interpretation)
    return {
        "short_mean": float(short_fare.mean()), "long_mean": float(long_fare.mean()),
        "t_statistic": float(statistic), "p_value": float(p_value),
        "interpretation": interpretation,
    }


def build_pipeline() -> Pipeline:
    """결측치 처리·스케일링·범주 인코딩과 분류 모델을 단일 Pipeline으로 구성한다."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    classifier = HistGradientBoostingClassifier(max_iter=120, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_evaluate_save(
    frame: pd.DataFrame, model_path: Path, max_training_rows: int
) -> dict[str, object]:
    """분류 Pipeline을 학습·예측·평가하고 joblib 모델을 저장한다."""
    model_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]
    model_frame = frame[model_columns]
    if len(model_frame) > max_training_rows:
        model_frame = model_frame.sample(max_training_rows, random_state=42)

    x = model_frame[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    y = model_frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    joblib.dump(pipeline, model_path)

    print("\n[ML Pipeline 평가]")
    print(f"학습·평가 표본={len(model_frame):,}, accuracy={accuracy:.6f}, F1={f1:.6f}")
    print(classification_report(y_test, predictions))
    print(f"모델 저장: {model_path}")
    return {
        "sample_rows": len(model_frame), "accuracy": float(accuracy), "f1": float(f1),
        "classification_report": report,
    }


def write_report(
    output_path: Path, source_name: str, quality: dict[str, object],
    descriptive: dict[str, object], t_test: dict[str, float | str],
    model: dict[str, object], long_trip_cutoff: float,
) -> None:
    """분석 결과와 산출물 위치를 발표용 Markdown 보고서로 자동 생성한다."""
    correlation = descriptive["correlation"]
    fare_distance_corr = correlation["trip_distance"]["fare_amount"]
    missing_text = ", ".join(
        f"{column}: {count:,}" for column, count in quality["pandas_missing"].items()
    ) or "없음"
    report = f"""# NYC Yellow Taxi End-to-End 분석 보고서

## 1. 프로젝트 개요

- 데이터: `{source_name}`
- 원본 크기: {quality['pandas_shape'][0]:,}행 × {quality['pandas_shape'][1]}열
- 목표: Pandas·Polars 비교부터 시각화, 통계 검정, ML Pipeline 저장까지 자동화
- 분류 목표: `trip_distance`가 정제 데이터 중앙값 {long_trip_cutoff:.3f} miles 이상인 장거리 운행

## 2. 데이터 준비 및 EDA

- Pandas/Polars 행·열 수 일치: {quality['pandas_shape'] == quality['polars_shape']}
- Pandas/Polars 평균 총액: {quality['pandas_total_mean']:.3f} / {quality['polars_total_mean']:.3f}
- 중복 행: Pandas {quality['pandas_duplicates']:,}, Polars {quality['polars_duplicates']:,}
- 주요 결측치: {missing_text}
- 비정상 거리·요금·소요시간 제거 후: {quality['clean_rows']:,}행
- 거리와 기본요금의 상관계수: {fare_distance_corr:.4f}

## 3. 시각화

- Seaborn 정적 차트: `seaborn_fare_distribution.png`
- Plotly 인터랙티브 차트: `plotly_hourly_payment.html`
- 두 차트 모두 제목과 축 레이블을 포함한다.

## 4. 통계 검정

- 검정: 장거리 vs 단거리 운행의 `fare_amount` Welch 독립표본 t-test
- 단거리 평균 요금: ${t_test['short_mean']:.3f}
- 장거리 평균 요금: ${t_test['long_mean']:.3f}
- t-statistic: {t_test['t_statistic']:.6f}
- p-value: {t_test['p_value']:.6g}
- 해석: {t_test['interpretation']}

## 5. ML Pipeline

- Pipeline: 수치 결측치 대체·표준화 + 범주 결측치 대체·인코딩 + HistGradientBoostingClassifier
- 평가 표본: {model['sample_rows']:,}행
- 정확도(Accuracy): {model['accuracy']:.4f}
- F1 score: {model['f1']:.4f}
- 저장 모델: `taxi_long_trip_pipeline.joblib`

## 6. 결론

Pandas와 Polars의 핵심 결과가 일치했으며, 시각화와 t-test를 통해 거리 그룹별 요금 차이를 확인했다.
저장된 Pipeline은 동일한 전처리와 모델을 함께 재사용할 수 있다.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """End-to-End 분석 전 단계를 실행하고 모든 산출물을 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="?", type=Path, default=Path("yellow_tripdata_2026-05.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("project_outputs"))
    parser.add_argument("--max-training-rows", type=int, default=500_000)
    args = parser.parse_args()

    try:
        if not args.parquet.is_file():
            raise FileNotFoundError(f"Parquet 파일을 찾을 수 없습니다: {args.parquet}")
        if args.max_training_rows < 10_000:
            raise ValueError("--max-training-rows는 10,000 이상이어야 합니다.")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        frame, quality = load_compare_clean(args.parquet)
        long_trip_cutoff = float(frame["trip_distance"].median())
        frame[TARGET] = frame["trip_distance"].ge(long_trip_cutoff).astype("int8")
        print("[Pandas·Polars 비교]")
        print(json.dumps(quality, ensure_ascii=False, indent=2))

        descriptive = descriptive_analysis(frame)
        create_seaborn_chart(frame, args.output_dir / "seaborn_fare_distribution.png")
        create_plotly_chart(frame, args.output_dir / "plotly_hourly_payment.html")
        t_test = run_t_test(frame)
        model = train_evaluate_save(
            frame, args.output_dir / "taxi_long_trip_pipeline.joblib", args.max_training_rows
        )
        write_report(
            args.output_dir / "report.md", args.parquet.name, quality, descriptive,
            t_test, model, long_trip_cutoff,
        )
        (args.output_dir / "metrics.json").write_text(
            json.dumps({"quality": quality, "t_test": t_test, "model": model}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n완료: 모든 산출물을 {args.output_dir}에 저장했습니다.")
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError, pl.exceptions.PolarsError) as error:
        raise SystemExit(f"오류: {error}") from error


if __name__ == "__main__":
    main()
