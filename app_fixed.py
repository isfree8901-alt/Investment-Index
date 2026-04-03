from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="US Macro Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START_DATE = "2005-01-01"
LOOKBACK_MONTHS = 36

SERIES_META = {
    "pmi": {
        "label": "ISM Manufacturing PMI",
        "source": "csv",
        "unit": "index",
        "freq": "monthly",
    },
    "housing": {
        "label": "Housing Starts",
        "series_id": "HOUST",
        "source": "fred",
        "unit": "k saar",
        "freq": "monthly",
    },
    "yield_spread": {
        "label": "10Y - 3M Treasury Spread",
        "series_id": "T10Y3M",
        "source": "fred",
        "unit": "%",
        "freq": "monthly",
    },
    "unemployment": {
        "label": "Unemployment Rate",
        "series_id": "UNRATE",
        "source": "fred",
        "unit": "%",
        "freq": "monthly",
    },
    "cpi": {
        "label": "Headline CPI",
        "series_id": "CPIAUCSL",
        "source": "fred",
        "unit": "index",
        "freq": "monthly",
    },
    "core_cpi": {
        "label": "Core CPI",
        "series_id": "CPILFESL",
        "source": "fred",
        "unit": "index",
        "freq": "monthly",
    },
}

ASSET_LABELS = {
    "equities": "주식",
    "treasuries": "채권",
    "gold": "금",
    "cash": "현금/MMF",
    "usd": "달러",
    "reits": "리츠",
}

ASSET_ORDER = ["equities", "treasuries", "gold", "cash", "usd", "reits"]


@dataclass
class IndicatorSignal:
    key: str
    label: str
    current: float
    previous: float
    change: float
    trend_3m: float
    score: int
    state: str
    summary: str
    reason: str
    market_impact: str
    threshold_note: str


def fetch_fred_series(series_id: str, api_key: str, start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    response = requests.get(FRED_API_BASE, params=params, timeout=30)

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"FRED 조회 실패 - series_id={series_id} - status={response.status_code} - detail={detail}"
        )

    payload = response.json()
    obs = payload.get("observations", [])
    if not obs:
        raise ValueError(f"No observations returned for {series_id}")

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=30 * 60)
def load_csv_series(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip().lower() for c in df.columns]
    if {"date", "value"}.difference(df.columns):
        raise ValueError("CSV must contain 'date' and 'value' columns.")
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=30 * 60)
def load_sample_csv(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return load_csv_series(f.read())


@st.cache_data(ttl=30 * 60)
def load_demo_expectations() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=60, freq="MS")
    values = [
        92, 89, 87, 85, 83, 80, 78, 76, 79, 82, 84, 81,
        79, 77, 75, 72, 70, 68, 71, 74, 76, 78, 77, 75,
        73, 72, 70, 68, 66, 64, 67, 69, 72, 74, 73, 71,
        69, 68, 66, 65, 64, 63, 65, 67, 68, 69, 71, 73,
        72, 70, 69, 68, 67, 66, 68, 70, 72, 74, 76, 75,
    ]
    return pd.DataFrame({"date": dates, "value": values})


@st.cache_data(ttl=30 * 60)
def load_demo_pmi() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=60, freq="MS")
    values = [
        58.7, 60.1, 61.4, 60.7, 59.2, 57.9, 56.8, 55.5, 54.4, 53.8, 52.7, 51.9,
        50.8, 49.9, 49.1, 48.3, 47.6, 47.1, 46.8, 47.4, 48.2, 49.0, 49.7, 50.2,
        50.8, 51.2, 50.5, 49.7, 49.1, 48.8, 48.4, 48.9, 49.5, 49.8, 50.4, 50.9,
        51.3, 50.6, 49.8, 49.2, 48.9, 48.5, 48.1, 47.8, 47.4, 47.9, 48.6, 49.1,
        49.7, 50.3, 50.7, 50.1, 49.5, 49.0, 48.7, 48.4, 48.8, 49.2, 49.8, 50.4,
    ]
    return pd.DataFrame({"date": dates, "value": values})


def latest(df: pd.DataFrame) -> Tuple[float, float, pd.Series]:
    now = float(df.iloc[-1]["value"])
    prev = float(df.iloc[-2]["value"]) if len(df) >= 2 else now
    return now, prev, df.iloc[-1]


def calc_trend_3m(df: pd.DataFrame) -> float:
    if len(df) < 4:
        return 0.0
    return float(df.iloc[-1]["value"] - df.iloc[-4]["value"])


def yoy_pct(df: pd.DataFrame) -> float:
    if len(df) < 13:
        return float("nan")
    current = float(df.iloc[-1]["value"])
    prior = float(df.iloc[-13]["value"])
    if prior == 0:
        return float("nan")
    return ((current / prior) - 1.0) * 100.0


def annual_inflation(df: pd.DataFrame) -> float:
    return yoy_pct(df)


def sahm_rule(df: pd.DataFrame) -> float:
    if len(df) < 12:
        return float("nan")
    rolling = df["value"].rolling(3).mean()
    current_3m = float(rolling.iloc[-1])
    trailing_low = float(rolling.iloc[-12:].min())
    return current_3m - trailing_low


def clip_score(value: float, min_v: float = 0, max_v: float = 100) -> float:
    return max(min_v, min(max_v, value))


def pmi_signal(df: pd.DataFrame) -> IndicatorSignal:
    current, previous, _ = latest(df)
    change = current - previous
    trend = calc_trend_3m(df)
    if current >= 50:
        score, state = 1, "확장"
    elif current >= 47:
        score, state = -1, "둔화 주의"
    else:
        score, state = -2, "위축 심화"

    summary = (
        "50 상회로 제조업 확장 신호가 유지됩니다."
        if current >= 50
        else "50 하회로 제조업 활동이 위축 국면에 있습니다."
    )
    reason = (
        "PMI는 신규주문·생산·고용·재고·공급망 응답을 반영하는 확산지수입니다. "
        "하락은 보통 주문 둔화와 생산 약화 가능성을 시사합니다."
    )
    impact = "PMI 약세는 경기민감주에 부담이고, 채권에는 우호적일 수 있습니다."
    return IndicatorSignal(
        key="pmi",
        label="ISM Manufacturing PMI",
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="공식 기준선 50 / 운영 경계선 47",
    )


def expectations_signal(df: pd.DataFrame) -> IndicatorSignal:
    current, previous, _ = latest(df)
    change = current - previous
    trend = calc_trend_3m(df)
    if current >= 80:
        score, state = 1, "소비 기대 안정"
    elif current >= 75:
        score, state = -1, "주의"
    else:
        score, state = -2, "강한 경고"
    summary = (
        "80 이상으로 단기 소비 기대가 상대적으로 안정적입니다."
        if current >= 80
        else "80 하회로 향후 소득·고용·경기 기대가 약화된 상태입니다."
    )
    reason = (
        "Conference Board Expectations Index는 향후 소득, 사업환경, 노동시장 전망을 반영합니다. "
        "하락은 소비 둔화 가능성과 경기 우려 확대로 해석할 수 있습니다."
    )
    impact = "기대지수 약세는 내수와 경기민감 자산에 부담이며, 방어자산 선호를 높일 수 있습니다."
    return IndicatorSignal(
        key="expectations",
        label="Consumer Expectations Index",
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="경고선 80 / 운영 강경고 75",
    )


def housing_signal(df: pd.DataFrame) -> IndicatorSignal:
    current, previous, _ = latest(df)
    change = current - previous
    trend = calc_trend_3m(df)
    yoy = yoy_pct(df)
    ma3 = float(df["value"].rolling(3).mean().iloc[-1])
    ma12 = float(df["value"].rolling(12).mean().iloc[-1]) if len(df) >= 12 else ma3

    if yoy >= 5:
        score, state = 1, "확장"
    elif yoy <= -10:
        score, state = -2, "경고"
    else:
        score, state = 0, "중립"

    if ma3 < ma12 and score > -2:
        score = min(score, -1)
        state = "주의"

    summary = f"전년동월비 {yoy:.1f}%로 주택 경기 흐름을 보여줍니다."
    reason = (
        "주택착공은 금리와 모기지 비용, 가계 구매력, 건설사 심리의 영향을 크게 받습니다. "
        "감소는 금리 부담 또는 수요 약화 신호일 수 있습니다."
    )
    impact = "주택착공 약세는 경기순환주와 리츠에 부담이 될 수 있습니다."
    return IndicatorSignal(
        key="housing",
        label="Housing Starts",
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="운영 기준: YoY +5 확장 / -10 이하 경고, 3MMA vs 12MMA 보조",
    )


def spread_signal(df: pd.DataFrame) -> IndicatorSignal:
    monthly = df.set_index("date").resample("MS").last().dropna().reset_index()
    current, previous, _ = latest(monthly)
    change = current - previous
    trend = calc_trend_3m(monthly)
    if current >= 1.0:
        score, state = 1, "정상 확장"
    elif current >= 0:
        score, state = -1, "둔화 주의"
    elif current >= -0.5:
        score, state = -2, "역전 경고"
    else:
        score, state = -2, "강한 경고"

    summary = (
        "장단기 금리차가 음수면 보통 경기침체 경고 신호로 해석됩니다."
        if current < 0
        else "장단기 금리차는 아직 정상 범위지만 둔화 여부를 함께 봐야 합니다."
    )
    reason = (
        "금리 스프레드는 장기 성장 기대와 단기 정책금리의 차이를 보여줍니다. "
        "축소·역전은 성장 기대 약화와 향후 금리인하 기대를 반영할 수 있습니다."
    )
    impact = "스프레드 축소·역전은 주식보다 채권과 방어자산 쪽에 우호적인 경우가 많습니다."
    return IndicatorSignal(
        key="yield_spread",
        label="10Y - 3M Spread",
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="핵심 기준선 0 / 운영 기준 +1.00, -0.50",
    )


def unemployment_signal(df: pd.DataFrame) -> IndicatorSignal:
    current, previous, _ = latest(df)
    change = current - previous
    trend = calc_trend_3m(df)
    sahm = sahm_rule(df)
    if sahm >= 0.5:
        score, state = -2, "침체 확인 경고"
    elif trend > 0.2:
        score, state = -1, "고용 둔화"
    else:
        score, state = 0, "안정"

    summary = f"실업률 {current:.1f}% / Sahm Rule {sahm:.2f}%p로 노동시장 냉각 여부를 확인합니다."
    reason = (
        "실업률은 경기 둔화가 실제 고용 약화로 이어졌는지 확인하는 후행지표입니다. "
        "Sahm Rule 상승은 경기침체 신호로 자주 활용됩니다."
    )
    impact = "실업률 상승은 주식에 부담이고, 경기방어 자산에는 우호적일 수 있습니다."
    return IndicatorSignal(
        key="unemployment",
        label="Unemployment Rate",
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="운영 기준: Sahm Rule 0.50%p 이상 경고",
    )


def cpi_signal(cpi_df: pd.DataFrame, core_df: pd.DataFrame) -> IndicatorSignal:
    current = annual_inflation(cpi_df)
    previous = annual_inflation(cpi_df.iloc[:-1].copy()) if len(cpi_df) > 13 else current
    change = current - previous
    trend = current - annual_inflation(cpi_df.iloc[:-3].copy()) if len(cpi_df) > 15 else 0.0
    core_now = annual_inflation(core_df)

    if max(current, core_now) >= 3.0:
        score, state = -1, "물가 부담"
    elif max(current, core_now) >= 2.0:
        score, state = 0, "완화 중"
    else:
        score, state = 1, "안정"

    summary = f"Headline CPI YoY {current:.1f}%, Core CPI YoY {core_now:.1f}%입니다."
    reason = (
        "CPI는 도시 소비자가 구매하는 재화·서비스 바스켓의 평균 가격 변화를 보여줍니다. "
        "특히 주거비와 서비스 물가가 하락 속도를 늦출 수 있습니다."
    )
    impact = "물가가 높으면 금리 인하 기대가 약해져 주식·리츠에 부담이 될 수 있습니다."
    return IndicatorSignal(
        key="cpi",
        label="CPI / Core CPI",
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="운영 기준: 3% 이상 부담 / 2~3% 완화 / 2% 부근 안정",
    )


def classify_regime(lead_score: float, lag_score: float, inflation_score: int) -> str:
    if lead_score <= -1.2 and lag_score <= -0.8 and inflation_score >= 0:
        return "침체 위험 확대"
    if lead_score <= -1.0 and inflation_score < 0:
        return "스태그플레이션 경계"
    if lead_score <= -0.5:
        return "성장 둔화"
    if lead_score > 0 and lag_score >= -0.2 and inflation_score >= 0:
        return "확장/회복"
    return "혼조/전환 구간"


def yoy_pct_state(summary: str) -> bool:
    if "전년동월비" not in summary:
        return False
    try:
        value_text = summary.split("전년동월비 ")[-1].split("%")[0]
        return float(value_text) >= 0
    except Exception:
        return False


def grade_from_score(score: float) -> str:
    if score >= 80:
        return "S"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def stance_from_score(score: float) -> str:
    if score >= 75:
        return "비중 확대 우호"
    if score >= 60:
        return "상대 우호"
    if score >= 45:
        return "중립"
    return "방어적 접근"


def asset_drivers(asset: str) -> str:
    bullet_map = {
        "equities": [
            "성장 지표 회복에 민감",
            "실업률 상승과 스프레드 역전에 취약",
            "물가 재가속 시 밸류에이션 부담",
        ],
        "treasuries": [
            "성장 둔화·실업률 상승에 우호적",
            "물가 둔화가 강할수록 유리",
            "수익률곡선 역전은 방어적 수요를 자극",
        ],
        "gold": [
            "정책·경기 불확실성에 강한 편",
            "물가 불안과 실질금리 하락 기대에 민감",
            "위기 방어 성격",
        ],
        "cash": [
            "방향성 불명확 구간에서 대기 자산 역할",
            "변동성 확대 대응용",
            "혼조 국면에서 유용",
        ],
        "usd": [
            "리스크오프 환경에서 상대 강세 가능",
            "미국 금리 우위에 민감",
            "글로벌 불확실성 확대 시 선호",
        ],
        "reits": [
            "금리와 주택 경기 흐름에 민감",
            "물가와 자금조달 비용 영향 큼",
            "경기 둔화 심화 시 부담",
        ],
    }
    return " / ".join(bullet_map[asset])


def compute_asset_scores(signals: Dict[str, IndicatorSignal]) -> pd.DataFrame:
    pmi = signals["pmi"]
    exp = signals["expectations"]
    housing = signals["housing"]
    spread = signals["yield_spread"]
    unemp = signals["unemployment"]
    cpi = signals["cpi"]

    pmi_support = 1 if pmi.current >= 50 else -1
    exp_support = 1 if exp.current >= 80 else -1
    housing_support = 1 if yoy_pct_state(signals["housing"].summary) else 0
    spread_support = 1 if spread.current > 0 else -1
    labor_soft = 1 if unemp.score < 0 else 0
    inflation_soft = 1 if cpi.current < 3 else -1

    base = {
        "equities": 50,
        "treasuries": 50,
        "gold": 50,
        "cash": 50,
        "usd": 50,
        "reits": 50,
    }

    base["equities"] += 12 * pmi_support + 10 * exp_support + 8 * spread_support + 5 * inflation_soft - 10 * labor_soft
    base["treasuries"] += -8 * pmi_support - 8 * exp_support - 10 * spread_support + 12 * labor_soft + 8 * inflation_soft
    base["gold"] += -2 * pmi_support - 3 * exp_support - 2 * spread_support + 6 * labor_soft - 8 * inflation_soft
    base["cash"] += -4 * pmi_support - 3 * exp_support - 4 * spread_support + 6 * labor_soft - 1 * inflation_soft
    base["usd"] += -3 * pmi_support - 2 * exp_support - 2 * spread_support + 5 * labor_soft - 3 * inflation_soft
    base["reits"] += 5 * pmi_support + 3 * exp_support + 4 * inflation_soft - 6 * labor_soft - 8 * (1 if spread.current > 1 else 0)

    if housing.score < 0:
        base["reits"] -= 8
        base["equities"] -= 3
        base["treasuries"] += 3
    elif housing.score > 0:
        base["reits"] += 6
        base["equities"] += 2

    if spread.current < 0:
        base["treasuries"] += 6
        base["gold"] += 4
        base["cash"] += 3
        base["equities"] -= 8
        base["reits"] -= 5

    if unemp.score <= -2:
        base["treasuries"] += 7
        base["gold"] += 4
        base["cash"] += 5
        base["equities"] -= 10
        base["reits"] -= 6

    if cpi.score < 0:
        base["treasuries"] -= 7
        base["gold"] += 8
        base["cash"] += 2
        base["reits"] -= 4
    elif cpi.score > 0:
        base["treasuries"] += 5
        base["equities"] += 3
        base["reits"] += 3

    rows = []
    for asset in ASSET_ORDER:
        score = round(clip_score(base[asset]))
        grade = grade_from_score(score)
        stance = stance_from_score(score)
        drivers = asset_drivers(asset)
        rows.append(
            {
                "asset_key": asset,
                "asset": ASSET_LABELS[asset],
                "score": score,
                "grade": grade,
                "stance": stance,
                "drivers": drivers,
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "asset"], ascending=[False, True]).reset_index(drop=True)


def build_combined_history(raw: Dict[str, pd.DataFrame], expectations: pd.DataFrame) -> pd.DataFrame:
    base = raw["pmi"].rename(columns={"value": "pmi"}).set_index("date")
    base = base.join(expectations.rename(columns={"value": "expectations"}).set_index("date"), how="outer")
    base = base.join(raw["housing"].rename(columns={"value": "housing"}).set_index("date"), how="outer")
    spread_monthly = raw["yield_spread"].set_index("date").resample("MS").last().rename(columns={"value": "yield_spread"})
    base = base.join(spread_monthly, how="outer")
    base = base.join(raw["unemployment"].rename(columns={"value": "unemployment"}).set_index("date"), how="outer")
    base = base.join(raw["cpi"].rename(columns={"value": "cpi"}).set_index("date"), how="outer")
    base = base.join(raw["core_cpi"].rename(columns={"value": "core_cpi"}).set_index("date"), how="outer")
    base = base.sort_index().tail(LOOKBACK_MONTHS)
    return base.reset_index()


def narrative_for_dashboard(regime: str, ranked_assets: pd.DataFrame, signals: Dict[str, IndicatorSignal]) -> str:
    top = ranked_assets.iloc[0]
    bottom = ranked_assets.iloc[-1]
    return (
        f"현재 국면은 **{regime}**으로 판정됩니다. "
        f"상대적으로 유리한 자산은 **{top['asset']}({top['score']}점)**, 상대적으로 부담이 큰 자산은 **{bottom['asset']}({bottom['score']}점)** 입니다. "
        f"핵심 근거는 PMI {signals['pmi'].current:.1f}, 기대지수 {signals['expectations'].current:.1f}, "
        f"장단기 스프레드 {signals['yield_spread'].current:.2f}, 실업률 {signals['unemployment'].current:.1f}%, "
        f"헤드라인 CPI YoY {signals['cpi'].current:.1f}%입니다."
    )


def contribution_table(signals: Dict[str, IndicatorSignal]) -> pd.DataFrame:
    items = [
        ("PMI", signals["pmi"].score, 25),
        ("기대지수", signals["expectations"].score, 20),
        ("주택착공", signals["housing"].score, 15),
        ("금리 스프레드", signals["yield_spread"].score, 25),
        ("실업률", signals["unemployment"].score, 10),
        ("CPI", signals["cpi"].score, 5),
    ]
    rows = []
    for label, score, weight in items:
        rows.append({"지표": label, "상태점수": score, "가중치": weight, "기여도": score * weight})
    return pd.DataFrame(rows)


def load_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("FRED_API_KEY", "")
        except Exception:
            key = ""
    return key


def metric_delta(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:.2f}"


def state_badge(state: str) -> str:
    if any(token in state for token in ["확장", "안정", "우호", "정상"]):
        return "🟢"
    if any(token in state for token in ["중립", "완화", "주의", "둔화"]):
        return "🟠"
    return "🔴"


def plot_series(df: pd.DataFrame, y: str, title: str, hline: Optional[float] = None) -> None:
    chart_df = df[["date", y]].dropna().copy()
    fig = px.line(chart_df, x="date", y=y, title=title)
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_signal_card(signal: IndicatorSignal, value_format: str = "{:.1f}") -> None:
    with st.container(border=True):
        st.markdown(f"### {signal.label}")
        st.metric(
            label=f"{state_badge(signal.state)} {signal.state}",
            value=value_format.format(signal.current),
            delta=metric_delta(signal.change),
        )
        st.caption(signal.threshold_note)
        st.write(signal.summary)
        st.write(f"**변화 이유**: {signal.reason}")
        st.write(f"**시장 영향**: {signal.market_impact}")


def render_asset_ranking(ranked_assets: pd.DataFrame) -> None:
    st.subheader("자산 랭킹")
    for idx, row in ranked_assets.iterrows():
        with st.container(border=True):
            left, right = st.columns([1.2, 2.5])
            with left:
                st.markdown(f"### {idx + 1}. {row['asset']}")
                st.metric("점수", f"{row['score']}점", row["grade"])
                st.caption(row["stance"])
            with right:
                st.progress(int(row["score"]))
                st.write(row["drivers"])


def build_downloadable_snapshot(ranked_assets: pd.DataFrame, signals: Dict[str, IndicatorSignal], regime: str) -> pd.DataFrame:
    rows = []
    for key in ["pmi", "expectations", "housing", "yield_spread", "unemployment", "cpi"]:
        s = signals[key]
        rows.append(
            {
                "type": "indicator",
                "name": s.label,
                "current": s.current,
                "state": s.state,
                "score": s.score,
                "summary": s.summary,
            }
        )
    for _, row in ranked_assets.iterrows():
        rows.append(
            {
                "type": "asset",
                "name": row["asset"],
                "current": row["score"],
                "state": row["stance"],
                "score": row["grade"],
                "summary": row["drivers"],
            }
        )
    rows.append({"type": "regime", "name": "regime", "current": np.nan, "state": regime, "score": np.nan, "summary": "macro regime"})
    return pd.DataFrame(rows)


def resolve_csv_input(uploaded_file, sample_path: str, use_sample: bool, demo_loader, label: str):
    try:
        if uploaded_file is not None:
            return load_csv_series(uploaded_file.getvalue()), "업로드된 CSV"
        if use_sample and os.path.exists(sample_path):
            return load_sample_csv(sample_path), "프로젝트 샘플 CSV"
        return demo_loader(), f"{label} 데모 데이터"
    except Exception as exc:
        raise RuntimeError(f"{label} 데이터 로딩 실패: {exc}") from exc


def main() -> None:
    st.title("미국 선행·후행지표 기반 자산배분 대시보드")
    st.caption(
        "Streamlit + FRED 기반. PMI와 Conference Board Expectations Index는 FRED 시리즈 제거/공개 제한 이슈를 고려해 CSV 업로드 또는 샘플/데모 데이터로 사용합니다."
    )

    with st.sidebar:
        st.header("설정")
        start_date = st.date_input("조회 시작일", pd.to_datetime(DEFAULT_START_DATE))
        fred_api_key = load_api_key()

        if fred_api_key:
            st.success("FRED API Key 인식됨")
        else:
            st.warning("FRED API Key가 없습니다. .streamlit/secrets.toml 또는 환경변수에 FRED_API_KEY를 넣어주세요.")

        st.subheader("PMI 입력")
        pmi_uploaded = st.file_uploader("PMI CSV 업로드", type=["csv"])
        use_sample_pmi = st.checkbox("PMI 샘플/데모 사용", value=True)

        st.subheader("Expectations Index 입력")
        expectations_uploaded = st.file_uploader("Conference Board Expectations CSV 업로드", type=["csv"])
        use_sample_expectations = st.checkbox("Expectations 샘플/데모 사용", value=True)

        show_debug = st.checkbox("디버그 데이터 표시", value=False)

    if not fred_api_key:
        st.error("FRED API Key 없이는 앱이 실행되지 않습니다.")
        st.stop()

    raw: Dict[str, pd.DataFrame] = {}

    try:
        for key, meta in SERIES_META.items():
            if meta.get("source") == "fred":
                raw[key] = fetch_fred_series(meta["series_id"], fred_api_key, str(start_date))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    pmi_sample_path = os.path.join(os.path.dirname(__file__), "data", "pmi_sample.csv")
    expectations_path = os.path.join(os.path.dirname(__file__), "data", "conference_board_expectations_sample.csv")

    try:
        pmi_df, pmi_source = resolve_csv_input(
            pmi_uploaded,
            pmi_sample_path,
            use_sample_pmi,
            load_demo_pmi,
            "PMI",
        )
        expectations_df, expectations_source = resolve_csv_input(
            expectations_uploaded,
            expectations_path,
            use_sample_expectations,
            load_demo_expectations,
            "Expectations Index",
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    raw["pmi"] = pmi_df

    signals = {
        "pmi": pmi_signal(raw["pmi"]),
        "expectations": expectations_signal(expectations_df),
        "housing": housing_signal(raw["housing"]),
        "yield_spread": spread_signal(raw["yield_spread"]),
        "unemployment": unemployment_signal(raw["unemployment"]),
        "cpi": cpi_signal(raw["cpi"], raw["core_cpi"]),
    }

    lead_score = np.average(
        [signals["pmi"].score, signals["expectations"].score, signals["housing"].score, signals["yield_spread"].score],
        weights=[25, 20, 15, 25],
    )
    lag_score = np.average([signals["unemployment"].score, signals["cpi"].score], weights=[10, 5])
    regime = classify_regime(lead_score, lag_score, signals["cpi"].score)

    ranked_assets = compute_asset_scores(signals)
    combined = build_combined_history(raw, expectations_df)
    summary_df = build_downloadable_snapshot(ranked_assets, signals, regime)

    top_left, top_mid, top_right = st.columns([1.2, 1.2, 2.2])
    with top_left:
        st.metric("현재 경기 국면", regime)
    with top_mid:
        risk_score = int(clip_score(50 + (-lead_score * 18) + (max(0, -lag_score) * 12)))
        st.metric("리스크 레벨", f"{risk_score}/100")
    with top_right:
        st.info(narrative_for_dashboard(regime, ranked_assets, signals))

    ranking_col, report_col = st.columns([1.3, 1.7])
    with ranking_col:
        render_asset_ranking(ranked_assets)
    with report_col:
        st.subheader("자동 해석 리포트")
        st.write(f"""
        - PMI: {signals['pmi'].summary}
        - 기대지수: {signals['expectations'].summary}
        - 주택착공: {signals['housing'].summary}
        - 금리 스프레드: {signals['yield_spread'].summary}
        - 실업률: {signals['unemployment'].summary}
        - CPI: {signals['cpi'].summary}
        """)
        st.markdown("#### 투자 방향 해석")
        st.write(
            f"상위 자산군은 **{', '.join(ranked_assets.head(3)['asset'].tolist())}** 입니다. "
            f"현재 국면이 {regime}로 판정되어 주식 편중 상태라면 상위 랭킹 자산을 활용한 분산 여부를 점검하는 구조로 해석할 수 있습니다."
        )
        st.caption(f"PMI 소스: {pmi_source} / Expectations Index 소스: {expectations_source}")

    st.divider()
    st.subheader("선행지표")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_signal_card(signals["pmi"], "{:.1f}")
    with c2:
        render_signal_card(signals["expectations"], "{:.1f}")
    with c3:
        render_signal_card(signals["housing"], "{:.0f}")
    with c4:
        render_signal_card(signals["yield_spread"], "{:.2f}")

    st.subheader("후행지표")
    c5, c6 = st.columns(2)
    with c5:
        render_signal_card(signals["unemployment"], "{:.1f}")
    with c6:
        render_signal_card(signals["cpi"], "{:.1f}")

    st.divider()
    tabs = st.tabs(["전체 요약", "선행지표 차트", "후행지표 차트", "자산 점수 근거", "데이터 다운로드"])

    with tabs[0]:
        st.dataframe(contribution_table(signals), use_container_width=True, hide_index=True)
        st.dataframe(ranked_assets[["asset", "score", "grade", "stance"]], use_container_width=True, hide_index=True)

    with tabs[1]:
        plot_series(combined, "pmi", "PMI", hline=50)
        plot_series(combined, "expectations", "Consumer Expectations Index", hline=80)
        housing_chart = combined[["date", "housing"]].dropna().copy()
        housing_chart["housing_yoy"] = housing_chart["housing"].pct_change(12) * 100
        plot_series(housing_chart, "housing_yoy", "Housing Starts YoY (%)", hline=0)
        plot_series(combined, "yield_spread", "10Y - 3M Spread", hline=0)

    with tabs[2]:
        plot_series(combined, "unemployment", "Unemployment Rate", hline=None)
        cpi_chart = combined[["date", "cpi", "core_cpi"]].dropna().copy()
        cpi_chart["headline_yoy"] = cpi_chart["cpi"].pct_change(12) * 100
        cpi_chart["core_yoy"] = cpi_chart["core_cpi"].pct_change(12) * 100
        plot_series(cpi_chart, "headline_yoy", "Headline CPI YoY (%)", hline=2)
        plot_series(cpi_chart, "core_yoy", "Core CPI YoY (%)", hline=2)

    with tabs[3]:
        st.markdown("#### 자산별 근거")
        for _, row in ranked_assets.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['asset']}** — {row['score']}점 ({row['grade']}, {row['stance']})")
                st.write(row["drivers"])

    with tabs[4]:
        csv = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("스냅샷 CSV 다운로드", data=csv, file_name="macro_dashboard_snapshot.csv", mime="text/csv")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if show_debug:
        st.divider()
        st.subheader("디버그")
        st.write({k: v.tail(3) for k, v in raw.items()})
        st.write(expectations_df.tail(3))


if __name__ == "__main__":
    main()
