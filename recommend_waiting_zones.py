"""현재 뉴욕 시각에 장거리 승객을 만날 가능성이 높은 대기 지역 Top 3 추천.

날짜×시간×승차 지역으로 과거 운행을 집계하고, 운행 전에 알 수 있는
요일·시간·승차 지역·과거 수요 평균만으로 전체 및 장거리 승차 건수를 예측한다.

변경내역:
- 2026-07-16: 현재 시각 기준 장거리 승객 대기지역 Top 3 추천 구현
- 2026-07-16: 미래 데이터 누수를 막는 시계열 5-fold 교차검증과 모델 설정 선택 추가
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import joblib
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError as error:
    raise SystemExit(
        f"필수 라이브러리가 없습니다: {error.name}\n"
        "설치: python -m pip install pandas polars pyarrow numpy plotly scikit-learn joblib"
    ) from error


NY_TZ = ZoneInfo("America/New_York")
FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "PULocationID",
    "zone_total_prior", "zone_long_prior", "zone_hour_total_prior", "zone_hour_long_prior",
]


def load_aggregate(parquet_path: Path, distance_threshold: float) -> pd.DataFrame:
    """원본 운행을 날짜×시간×승차 지역의 전체·장거리 수요로 집계한다."""
    source = pl.scan_parquet(parquet_path).select(
        "tpep_pickup_datetime", "PULocationID", "trip_distance"
    )
    month_match = re.search(r"(\d{4})-(\d{2})", parquet_path.stem)
    if month_match:
        analysis_year, analysis_month = map(int, month_match.groups())
        source = source.filter(
            (pl.col("tpep_pickup_datetime").dt.year() == analysis_year)
            & (pl.col("tpep_pickup_datetime").dt.month() == analysis_month)
        )
    distance_upper = source.select(
        pl.col("trip_distance").quantile(0.999)
    ).collect().item()
    grouped = (
        source.filter(
            pl.col("tpep_pickup_datetime").is_not_null()
            & pl.col("PULocationID").is_between(1, 265)
            & pl.col("trip_distance").is_between(0, distance_upper)
        )
        .with_columns(
            pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("hour"),
            (pl.col("trip_distance") >= distance_threshold).cast(pl.Int8).alias("is_long_trip"),
        )
        .group_by("pickup_date", "hour", "PULocationID")
        .agg(
            pl.len().alias("total_pickups"),
            pl.col("is_long_trip").sum().alias("long_trip_count"),
        )
        .collect()
        .to_pandas()
    )
    if grouped.empty:
        raise ValueError("정제 후 집계할 운행 데이터가 없습니다.")
    grouped["pickup_date"] = pd.to_datetime(grouped["pickup_date"])

    # 운행이 없었던 시간·지역도 실제 수요 0으로 학습하도록 완전한 격자를 만든다.
    dates = pd.DataFrame({"pickup_date": pd.date_range(grouped["pickup_date"].min(), grouped["pickup_date"].max())})
    hours = pd.DataFrame({"hour": range(24)})
    zones = pd.DataFrame({"PULocationID": sorted(grouped["PULocationID"].unique())})
    grid = dates.merge(hours, how="cross").merge(zones, how="cross")
    hourly = grid.merge(grouped, on=["pickup_date", "hour", "PULocationID"], how="left")
    hourly[["total_pickups", "long_trip_count"]] = hourly[
        ["total_pickups", "long_trip_count"]
    ].fillna(0)
    hourly["day"] = hourly["pickup_date"].dt.day
    hourly["day_of_week"] = hourly["pickup_date"].dt.dayofweek
    hourly["is_weekend"] = hourly["day_of_week"].ge(5).astype("int8")
    return hourly


def build_priors(history: pd.DataFrame) -> dict[str, object]:
    """학습 기간에서만 지역 및 지역×시간 과거 평균을 계산한다."""
    zone = history.groupby("PULocationID")[["total_pickups", "long_trip_count"]].mean()
    zone_hour = history.groupby(["PULocationID", "hour"])[["total_pickups", "long_trip_count"]].mean()
    return {
        "zone_total": zone["total_pickups"].to_dict(),
        "zone_long": zone["long_trip_count"].to_dict(),
        "zone_hour_total": zone_hour["total_pickups"].to_dict(),
        "zone_hour_long": zone_hour["long_trip_count"].to_dict(),
        "global_total": float(history["total_pickups"].mean()),
        "global_long": float(history["long_trip_count"].mean()),
    }


def make_features(frame: pd.DataFrame, priors: dict[str, object]) -> pd.DataFrame:
    """시간 순환 특성과 과거 수요 평균을 누수 없이 결합한다."""
    result = frame.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["zone_total_prior"] = result["PULocationID"].map(priors["zone_total"]).fillna(priors["global_total"])
    result["zone_long_prior"] = result["PULocationID"].map(priors["zone_long"]).fillna(priors["global_long"])
    keys = pd.Series(list(zip(result["PULocationID"], result["hour"])), index=result.index)
    result["zone_hour_total_prior"] = keys.map(priors["zone_hour_total"]).fillna(result["zone_total_prior"])
    result["zone_hour_long_prior"] = keys.map(priors["zone_hour_long"]).fillna(result["zone_long_prior"])
    return result


def build_model(parameters: dict[str, float | int] | None = None) -> Pipeline:
    """결측치 처리와 수요 회귀 모델을 하나의 Pipeline으로 구성한다."""
    model_parameters = {
        "max_iter": 140, "learning_rate": 0.07, "l2_regularization": 1.0,
        "max_leaf_nodes": 31, "random_state": 42,
    }
    if parameters:
        model_parameters.update(parameters)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(**model_parameters)),
    ])


def fit_models(
    history: pd.DataFrame, parameters: dict[str, float | int] | None = None
) -> tuple[Pipeline, Pipeline, dict[str, object]]:
    """전체 승차와 장거리 승차 건수를 예측하는 두 Pipeline을 학습한다."""
    priors = build_priors(history)
    featured = make_features(history, priors)
    total_model = build_model(parameters)
    long_model = build_model(parameters)
    total_model.fit(featured[FEATURES], featured["total_pickups"])
    long_model.fit(featured[FEATURES], featured["long_trip_count"])
    return total_model, long_model, priors


def predict_demand(
    frame: pd.DataFrame, total_model: Pipeline, long_model: Pipeline, priors: dict[str, object]
) -> pd.DataFrame:
    """전체·장거리 수요를 예측하고 음수 및 논리적으로 불가능한 값을 보정한다."""
    featured = make_features(frame, priors)
    result = frame.copy()
    result["predicted_total"] = np.maximum(total_model.predict(featured[FEATURES]), 0)
    predicted_long = np.maximum(long_model.predict(featured[FEATURES]), 0)
    result["predicted_long"] = np.minimum(predicted_long, result["predicted_total"])
    result["predicted_long_rate"] = np.divide(
        result["predicted_long"], result["predicted_total"],
        out=np.zeros(len(result), dtype=float), where=result["predicted_total"].gt(0),
    )
    return result


def precision_at_k(predictions: pd.DataFrame, k: int = 3) -> float:
    """각 시간의 실제 장거리 수요 Top K와 추천 Top K의 평균 일치 비율을 계산한다."""
    scores: list[float] = []
    for _, group in predictions.groupby(["pickup_date", "hour"]):
        predicted = set(group.nlargest(k, "predicted_long")["PULocationID"])
        actual = set(group.nlargest(k, "long_trip_count")["PULocationID"])
        scores.append(len(predicted & actual) / k)
    return float(np.mean(scores))


def evaluate_split(
    train: pd.DataFrame, test: pd.DataFrame, label: str,
    parameters: dict[str, float | int] | None = None,
) -> dict[str, float]:
    """시간 순서 분할에서 수요 오차와 추천 Precision@3를 평가한다."""
    total_model, long_model, priors = fit_models(train, parameters)
    predicted = predict_demand(test, total_model, long_model, priors)
    metrics = {
        "total_mae": float(mean_absolute_error(predicted["total_pickups"], predicted["predicted_total"])),
        "total_rmse": float(np.sqrt(mean_squared_error(predicted["total_pickups"], predicted["predicted_total"]))),
        "long_mae": float(mean_absolute_error(predicted["long_trip_count"], predicted["predicted_long"])),
        "long_rmse": float(np.sqrt(mean_squared_error(predicted["long_trip_count"], predicted["predicted_long"]))),
        "precision_at_3": precision_at_k(predicted, 3),
    }
    print(f"[{label}] " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    return metrics


def tune_with_time_series_5fold(history: pd.DataFrame) -> tuple[dict[str, float | int], dict[str, object]]:
    """시간 순서를 유지한 5-fold 교차검증으로 장거리 수요 MAE가 가장 낮은 설정을 고른다.

    일반 KFold처럼 데이터를 무작위로 섞지 않는다. 각 fold는 과거 날짜로 학습하고
    그 이후 날짜로 검증하므로 실제 미래 수요 예측 상황과 같은 방향을 유지한다.
    """
    candidates: list[dict[str, float | int]] = [
        {"max_iter": 100, "learning_rate": 0.08, "l2_regularization": 1.0, "max_leaf_nodes": 31},
        {"max_iter": 160, "learning_rate": 0.05, "l2_regularization": 2.0, "max_leaf_nodes": 31},
        {"max_iter": 140, "learning_rate": 0.06, "l2_regularization": 1.0, "max_leaf_nodes": 63},
    ]
    dates = np.array(sorted(history["pickup_date"].unique()))
    splitter = TimeSeriesSplit(n_splits=5)
    candidate_results: list[dict[str, object]] = []

    for candidate_number, parameters in enumerate(candidates, start=1):
        fold_metrics: list[dict[str, float]] = []
        for fold_number, (train_indices, validation_indices) in enumerate(splitter.split(dates), start=1):
            train_dates = dates[train_indices]
            validation_dates = dates[validation_indices]
            fold_train = history[history["pickup_date"].isin(train_dates)]
            fold_validation = history[history["pickup_date"].isin(validation_dates)]
            metrics = evaluate_split(
                fold_train, fold_validation,
                f"5-fold 후보 {candidate_number} / fold {fold_number}", parameters,
            )
            fold_metrics.append(metrics)

        mean_metrics = {
            key: float(np.mean([fold[key] for fold in fold_metrics]))
            for key in fold_metrics[0]
        }
        candidate_results.append({
            "parameters": parameters, "mean_metrics": mean_metrics, "folds": fold_metrics,
        })
        print(
            f"[후보 {candidate_number} 5-fold 평균] "
            f"long_mae={mean_metrics['long_mae']:.4f}, "
            f"precision_at_3={mean_metrics['precision_at_3']:.4f}"
        )

    best = min(candidate_results, key=lambda result: result["mean_metrics"]["long_mae"])
    print(f"[5-fold 최적 설정] {best['parameters']}")
    return best["parameters"], {
        "n_splits": 5,
        "strategy": "TimeSeriesSplit (expanding window, no shuffle)",
        "selection_metric": "mean long_trip_count MAE",
        "best_parameters": best["parameters"],
        "best_mean_metrics": best["mean_metrics"],
        "candidates": candidate_results,
    }


def parse_prediction_time(value: str | None) -> datetime:
    """입력 시각을 뉴욕 현지 시각으로 해석하며, 미입력 시 현재 뉴욕 시각을 사용한다."""
    if value is None:
        return datetime.now(NY_TZ).replace(minute=0, second=0, microsecond=0)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NY_TZ)
    return parsed.astimezone(NY_TZ).replace(minute=0, second=0, microsecond=0)


def recommend(
    prediction_time: datetime, zones: list[int], total_model: Pipeline,
    long_model: Pipeline, priors: dict[str, object], zone_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """지정 시각의 모든 승차 지역을 예측하고 예상 장거리 건수 기준으로 순위를 만든다."""
    candidates = pd.DataFrame({"PULocationID": zones})
    candidates["hour"] = prediction_time.hour
    candidates["day_of_week"] = prediction_time.weekday()
    candidates["is_weekend"] = int(prediction_time.weekday() >= 5)
    predicted = predict_demand(candidates, total_model, long_model, priors)
    predicted = predicted.merge(
        zone_lookup[["LocationID", "Borough", "Zone"]],
        left_on="PULocationID", right_on="LocationID", how="left",
    )
    predicted["Borough"] = predicted["Borough"].fillna("Unknown")
    predicted["Zone"] = predicted["Zone"].fillna("Zone " + predicted["PULocationID"].astype(str))
    predicted["recommendation_score"] = predicted["predicted_long"]
    return predicted.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


def create_dashboard(
    ranking: pd.DataFrame, prediction_time: datetime, output_path: Path,
    distance_threshold: float, metrics: dict[str, object],
) -> None:
    """Top 3 추천 카드와 상위 지역 비교 차트를 포함한 결과 HTML을 생성한다."""
    top3 = ranking.head(3)
    colors = ["#f5b700", "#9aa4b2", "#c67c3b"]
    cards = []
    for index, (_, row) in enumerate(top3.iterrows(), start=1):
        cards.append(f"""
        <article class="card rank-{index}">
          <div class="rank">추천 {index}위</div>
          <h2>{row['Zone']}</h2><p class="borough">{row['Borough']} · Zone {int(row['PULocationID'])}</p>
          <div class="metric"><b>{row['predicted_long']:.1f}</b><span>예상 장거리 승차/시간</span></div>
          <div class="sub"><span>전체 승차 {row['predicted_total']:.1f}건</span><span>장거리 비율 {row['predicted_long_rate']:.1%}</span></div>
        </article>""")

    chart_frame = ranking.head(10).copy()
    chart_frame["label"] = chart_frame["Zone"] + " (" + chart_frame["Borough"] + ")"
    chart = px.bar(
        chart_frame.sort_values("predicted_long"), x="predicted_long", y="label",
        orientation="h", color="predicted_long_rate", color_continuous_scale="Blues",
        hover_data={"predicted_total": ":.1f", "predicted_long_rate": ":.1%"},
        labels={"predicted_long": "예상 장거리 승차/시간", "label": "대기 지역", "predicted_long_rate": "장거리 비율"},
        title="장거리 수요 예상 상위 10개 지역",
    )
    chart.update_layout(height=560, margin=dict(l=20, r=20, t=60, b=20))
    plot_html = chart.to_html(full_html=False, include_plotlyjs=True)
    weekday = ["월", "화", "수", "목", "금", "토", "일"][prediction_time.weekday()]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>NYC 장거리 승객 대기지역 추천</title>
    <style>
      body{{margin:0;background:#f3f6fa;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif}}
      main{{max-width:1180px;margin:auto;padding:38px 24px}} header{{margin-bottom:25px}}
      h1{{margin:0 0 8px;font-size:34px}} .time{{font-size:18px;color:#526079}} .notice{{color:#65738a;font-size:14px}}
      .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}} .card{{background:white;border-radius:18px;padding:24px;box-shadow:0 8px 26px #1c33551a;border-top:7px solid}}
      .rank-1{{border-color:{colors[0]}}}.rank-2{{border-color:{colors[1]}}}.rank-3{{border-color:{colors[2]}}}
      .rank{{font-weight:800;color:#506079}} h2{{font-size:24px;margin:8px 0 2px}} .borough{{margin:0;color:#77849a}}
      .metric{{margin:24px 0 18px}} .metric b{{font-size:38px;display:block;color:#1261a6}} .metric span{{color:#617087}}
      .sub{{display:flex;justify-content:space-between;border-top:1px solid #e4e9f1;padding-top:14px;font-size:14px}}
      .panel{{background:white;margin-top:22px;border-radius:18px;padding:14px;box-shadow:0 8px 26px #1c335512}}
      @media(max-width:800px){{.cards{{grid-template-columns:1fr}}}}
    </style></head><body><main><header><h1>🚕 장거리 승객 대기지역 추천</h1>
    <div class="time">예측 시각: {prediction_time:%Y-%m-%d} ({weekday}) {prediction_time:%H}:00 · America/New_York</div>
    <p class="notice">장거리 기준 {distance_threshold:g} miles · 예상 장거리 승차 건수가 많은 순서 · 날씨/공급/실시간 교통은 미반영</p></header>
    <section class="cards">{''.join(cards)}</section><section class="panel">{plot_html}</section>
    <p class="notice">검증 Precision@3: {metrics['test']['precision_at_3']:.1%} · 과거 한 달 패턴 기반이므로 최신 데이터 추가 시 재학습 권장</p>
    </main></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    """시간순 검증·최종학습·현재 시각 추천·대시보드 저장을 수행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="?", type=Path, default=Path("yellow_tripdata_2026-05.parquet"))
    parser.add_argument("--zones", type=Path, default=Path("taxi_zone_lookup.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("recommendation_outputs"))
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--at", help='뉴욕 현지 예측 시각. 예: "2026-07-17 22:00"')
    args = parser.parse_args()

    try:
        if not args.parquet.is_file() or not args.zones.is_file():
            raise FileNotFoundError("Parquet 또는 taxi_zone_lookup.csv 파일을 찾을 수 없습니다.")
        if args.distance_threshold <= 0:
            raise ValueError("장거리 기준은 0보다 커야 합니다.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        hourly = load_aggregate(args.parquet, args.distance_threshold)

        # 5월 1~25일 안에서 시계열 5-fold로 설정을 선택하고, 26~31일은 최종 테스트로 보존한다.
        test_train = hourly[hourly["day"].le(25)]
        test = hourly[hourly["day"].ge(26)]
        best_parameters, cross_validation = tune_with_time_series_5fold(test_train)
        metrics = {
            "cross_validation": cross_validation,
            "test": evaluate_split(
                test_train, test, "최종 테스트 5/26~31", best_parameters
            ),
        }

        total_model, long_model, priors = fit_models(hourly, best_parameters)
        model_bundle = {
            "total_model": total_model, "long_model": long_model, "priors": priors,
            "features": FEATURES, "distance_threshold": args.distance_threshold,
            "cross_validation": cross_validation,
        }
        joblib.dump(model_bundle, args.output_dir / "waiting_zone_demand_models.joblib")

        prediction_time = parse_prediction_time(args.at)
        zone_lookup = pd.read_csv(args.zones)
        ranking = recommend(
            prediction_time, sorted(hourly["PULocationID"].unique()),
            total_model, long_model, priors, zone_lookup,
        )
        ranking.to_csv(args.output_dir / "current_zone_ranking.csv", index=False, encoding="utf-8-sig")
        create_dashboard(
            ranking, prediction_time, args.output_dir / "current_recommendations.html",
            args.distance_threshold, metrics,
        )
        (args.output_dir / "recommendation_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n[현재 시각 Top 3 추천]")
        print(ranking.head(3)[[
            "Borough", "Zone", "predicted_total", "predicted_long", "predicted_long_rate"
        ]].to_string(index=False, formatters={"predicted_long_rate": "{:.1%}".format}))
        print(f"\n결과창: {args.output_dir / 'current_recommendations.html'}")
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"오류: {error}") from error


if __name__ == "__main__":
    main()
