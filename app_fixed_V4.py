from __future__ import annotations

import html
import io
import math
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


TRADING_ECONOMICS_PMI_URL = "https://tradingeconomics.com/united-states/business-confidence"
CONFERENCE_BOARD_CCI_URL = "https://www.conference-board.org/topics/consumer-confidence/index.cfm"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9,ko-KR;q=0.8",
}

st.set_page_config(
    page_title="US Macro Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START_DATE = "2005-01-01"
ISM_PMI_CURRENT_URL = "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/"
CONFERENCE_BOARD_CONFIDENCE_URL = "https://www.conference-board.org/topics/consumer-confidence/index.cfm"

UPDATE_CYCLE_NOTES = {
    "pmi": "업데이트: 매월 1회",
    "expectations": "업데이트: 매월 1회",
    "housing": "업데이트: 매월 1회",
    "yield_spread": "업데이트: 일별",
    "unemployment": "업데이트: 매월 1회",
    "cpi": "업데이트: 매월 1회",
}

LOOKBACK_MONTHS = 36
BLOOMBERG_RSS_URL = "https://feeds.bloomberg.com/markets/news.rss"
TRANSLATE_API_URL = "https://translate.googleapis.com/translate_a/single"
DASHBOARD_COLORS = {
    "pink": "#F9B2D7",
    "blue": "#CFECF3",
    "green": "#DAF9DE",
    "yellow": "#F6FFDC",
    "text": "#25313C",
    "muted": "#5B6770",
    "border": "#D7E3E8",
    "danger": "#E95E7A",
    "warn": "#E6A93B",
    "success": "#53A971",
    "navy": "#5B6CFF",
}
CHART_SEQUENCE = [
    DASHBOARD_COLORS["pink"],
    DASHBOARD_COLORS["blue"],
    DASHBOARD_COLORS["green"],
    DASHBOARD_COLORS["yellow"],
    DASHBOARD_COLORS["navy"],
]

SERIES_META = {
    "pmi": {
        "label": "ISM 제조업 구매관리자지수 (PMI)",
        "short_label": "제조업 PMI",
        "source": "csv",
        "unit": "index",
        "freq": "monthly",
        "kind": "선행지표",
    },
    "housing": {
        "label": "주택착공건수 (Housing Starts)",
        "short_label": "주택착공",
        "series_id": "HOUST",
        "source": "fred",
        "unit": "k saar",
        "freq": "monthly",
        "kind": "선행지표",
    },
    "yield_spread": {
        "label": "미국채 10년-3개월 금리차 (10Y-3M Treasury Spread)",
        "short_label": "장단기 금리차",
        "series_id": "T10Y3M",
        "source": "fred",
        "unit": "%",
        "freq": "monthly",
        "kind": "선행지표",
    },
    "unemployment": {
        "label": "실업률 (Unemployment Rate)",
        "short_label": "실업률",
        "series_id": "UNRATE",
        "source": "fred",
        "unit": "%",
        "freq": "monthly",
        "kind": "후행지표",
    },
    "cpi": {
        "label": "소비자물가지수 (CPI)",
        "short_label": "CPI",
        "series_id": "CPIAUCSL",
        "source": "fred",
        "unit": "index",
        "freq": "monthly",
        "kind": "후행지표",
    },
    "core_cpi": {
        "label": "근원 소비자물가지수 (Core CPI)",
        "short_label": "근원 CPI",
        "series_id": "CPILFESL",
        "source": "fred",
        "unit": "index",
        "freq": "monthly",
        "kind": "후행지표",
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
    buy_idea: str


def inject_global_style() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --pink: {DASHBOARD_COLORS['pink']};
            --blue: {DASHBOARD_COLORS['blue']};
            --green: {DASHBOARD_COLORS['green']};
            --yellow: {DASHBOARD_COLORS['yellow']};
            --text: {DASHBOARD_COLORS['text']};
            --muted: {DASHBOARD_COLORS['muted']};
            --border: {DASHBOARD_COLORS['border']};
            --danger: {DASHBOARD_COLORS['danger']};
            --warn: {DASHBOARD_COLORS['warn']};
            --success: {DASHBOARD_COLORS['success']};
            --navy: {DASHBOARD_COLORS['navy']};
        }}

        .stApp {{
            background: linear-gradient(180deg, rgba(249,178,215,0.16) 0%, rgba(207,236,243,0.18) 32%, rgba(246,255,220,0.18) 68%, rgba(218,249,222,0.24) 100%);
            color: var(--text);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 14px 16px;
            box-shadow: 0 12px 24px rgba(60, 80, 90, 0.06);
        }}

        div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] {{
            border-radius: 22px;
        }}

        .hero-card {{
            background: linear-gradient(135deg, rgba(249,178,215,0.95) 0%, rgba(207,236,243,0.92) 55%, rgba(218,249,222,0.96) 100%);
            border: 1px solid rgba(255,255,255,0.65);
            border-radius: 28px;
            padding: 22px 24px;
            box-shadow: 0 16px 36px rgba(60, 80, 90, 0.10);
            margin-bottom: 12px;
        }}

        .hero-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .hero-regime {{
            font-size: 2rem;
            font-weight: 800;
            color: var(--text);
            line-height: 1.15;
            margin-bottom: 10px;
        }}

        .hero-desc {{
            font-size: 1rem;
            color: var(--text);
            line-height: 1.65;
        }}

        .insight-chip-wrap {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 16px;
        }}

        .insight-chip {{
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(91,108,255,0.14);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.9rem;
            font-weight: 600;
        }}

        .section-note {{
            background: rgba(255,255,255,0.66);
            border-left: 6px solid var(--navy);
            border-radius: 18px;
            padding: 14px 16px;
            margin: 6px 0 16px 0;
            color: var(--text);
        }}

        .signal-card {{
            background: rgba(255,255,255,0.80);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 18px;
            box-shadow: 0 12px 20px rgba(60, 80, 90, 0.05);
            min-height: 280px;
        }}

        .signal-kind {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(91,108,255,0.08);
            color: var(--navy);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .asset-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}

        .asset-card {{
            background: rgba(255,255,255,0.84);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 18px 18px 16px;
            box-shadow: 0 12px 20px rgba(60, 80, 90, 0.06);
        }}

        .asset-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 12px;
        }}

        .asset-rank {{
            font-size: 0.82rem;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .asset-name {{
            font-size: 1.28rem;
            font-weight: 800;
            color: var(--text);
            line-height: 1.2;
        }}

        .asset-grade {{
            background: rgba(91,108,255,0.10);
            color: var(--navy);
            border-radius: 14px;
            padding: 8px 10px;
            font-weight: 800;
            min-width: 60px;
            text-align: center;
        }}

        .asset-score {{
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 4px;
        }}

        .asset-stance {{
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .asset-bar {{
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: rgba(91,108,255,0.09);
            overflow: hidden;
            margin-bottom: 12px;
        }}

        .asset-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--pink) 0%, var(--blue) 45%, var(--green) 100%);
        }}

        .asset-drivers {{
            color: var(--text);
            font-size: 0.92rem;
            line-height: 1.55;
        }}

        .headline-card {{
            background: rgba(255,255,255,0.84);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 18px 20px;
            box-shadow: 0 12px 20px rgba(60, 80, 90, 0.05);
        }}

        .headline-row {{
            padding: 10px 0;
            border-bottom: 1px dashed rgba(91,108,255,0.16);
        }}

        .headline-row:last-child {{
            border-bottom: 0;
        }}

        .headline-meta {{
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 3px;
        }}

        .headline-title {{
            font-size: 0.98rem;
            color: var(--text);
            line-height: 1.55;
            font-weight: 600;
        }}

        @media (max-width: 980px) {{
            .asset-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def translate_text_to_korean(text: str) -> str:
    """
    간단 번역 함수
    - 외부 API 없이 기본 동작
    - 실패 시 원문 그대로 반환
    """
    try:
        # 무료 번역 (Google unofficial endpoint)
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "ko",
            "dt": "t",
            "q": text,
        }

        res = requests.get(url, params=params, timeout=5)
        if res.status_code != 200:
            return text

        data = res.json()

        # 번역 결과 조합
        translated = "".join([item[0] for item in data[0]])
        return translated

    except Exception:
        return text


@st.cache_data(ttl=60 * 30)
def fetch_te_pmi_reference() -> dict:
    try:
        res = requests.get(
            TRADING_ECONOMICS_PMI_URL,
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        res.raise_for_status()
        text = html.unescape(res.text)

        # 예: "Business Confidence in the United States increased to 52.70 points in March..."
        match = re.search(
            r"Business Confidence in the United States .*? to ([0-9]+(?:\.[0-9]+)?) points in ([A-Za-z]+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if not match:
            return {}

        value = float(match.group(1))
        period = match.group(2)

        return {
            "label": "미국 ISM 제조업 PMI",
            "value": value,
            "period": period,
            "unit": "points",
            "update_cycle": "매월 1회",
            "source": "Trading Economics",
            "link": TRADING_ECONOMICS_PMI_URL,
        }
    except Exception:
        return {}


def render_reference_panel() -> None:
    pmi_data = fetch_te_pmi_reference()
    exp_data = fetch_latest_expectations_public()

    st.subheader("자동 조회 참고")
    st.caption("공식 공개 페이지 기준 참고값입니다. 필요하면 아래 수동 입력란에 바로 반영해 주세요.")

    with st.container(border=True):
        st.markdown("#### 1) PMI")
        if pmi_data:
            st.metric("PMI 참고값", f"{pmi_data['value']:.1f}")
            st.caption(f"대상 월: {pmi_data.get('period', '-')}")
            st.caption("업데이트: 매월 1회")
            st.link_button("PMI 원문 보기", pmi_data["link"], use_container_width=True)
            st.code(
                f"{pd.Timestamp.today().strftime('%Y-%m-01')},{pmi_data['value']}",
                language="text",
            )
        else:
            st.warning("PMI 자동 조회 실패")
            st.link_button("PMI 원문 보기", TRADING_ECONOMICS_PMI_URL, use_container_width=True)

        st.divider()

        st.markdown("#### 2) 소비자 기대지수 (CBEI)")
        if exp_data and exp_data.get("value") is not None:
            st.metric("기대지수 참고값", f"{exp_data['value']:.1f}")
            if exp_data.get("cci_value") is not None:
                st.caption(f"전체 CCI: {exp_data['cci_value']:.1f}")
            if exp_data.get("present_situation_value") is not None:
                st.caption(f"현재상황지수: {exp_data['present_situation_value']:.1f}")
            if exp_data.get("release_date"):
                st.caption(f"발표일: {exp_data['release_date']}")
            st.caption(f"업데이트: {exp_data.get('update_cycle', '매월 1회')}")
            st.caption("※ 경기판단 기준으로 쓰는 값은 전체 CCI가 아니라 기대지수입니다.")
            st.link_button("Conference Board 원문 보기", exp_data["link"], use_container_width=True)
            st.code(
                f"{pd.Timestamp.today().strftime('%Y-%m-01')},{exp_data['value']}",
                language="text",
            )
            with st.expander("CBEI 디버그 보기", expanded=False):
                st.write("매칭 문장:", exp_data.get("matched_sentence", ""))
                st.write("본문 스니펫:", exp_data.get("debug_snippet", ""))
        else:
            st.warning("소비자 기대지수 자동 조회 실패")
            if exp_data and exp_data.get("error"):
                st.caption(f"오류: {exp_data['error']}")
            st.link_button("Conference Board 원문 보기", CONFERENCE_BOARD_CCI_URL, use_container_width=True)
            with st.expander("CBEI 디버그 보기", expanded=False):
                if exp_data:
                    st.write("매칭 문장:", exp_data.get("matched_sentence", ""))
                    st.write("본문 스니펫:", exp_data.get("debug_snippet", ""))



def fetch_fred_series(series_id: str, api_key: str, start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    response = requests.get(FRED_API_BASE, params=params, timeout=30, headers=REQUEST_HEADERS)

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


def build_manual_series_input(
    series_name: str,
    base_df: pd.DataFrame,
    latest_auto_value: Optional[float] = None,
    latest_auto_label: str = "자동 조회 최신값",
) -> pd.DataFrame:
    base_df = base_df[["date", "value"]].copy()
    base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
    base_df["value"] = pd.to_numeric(base_df["value"], errors="coerce")
    base_df = base_df.dropna().sort_values("date").reset_index(drop=True)

    if base_df.empty:
        return base_df

    latest_row = base_df.iloc[-1]
    latest_month = pd.to_datetime(latest_row["date"]).to_period("M").to_timestamp()
    current_value = float(latest_row["value"])
    default_value = float(latest_auto_value) if latest_auto_value is not None else current_value

    st.caption(f"{series_name} 최신 월 값을 직접 입력하면 즉시 전체 계산에 반영됩니다.")
    input_col1, input_col2 = st.columns([1.1, 1.2])
    with input_col1:
        manual_month = st.date_input(
            f"{series_name} 기준 월",
            value=latest_month.date(),
            key=f"{series_name}_manual_month",
            help="해당 월의 값이 기존 데이터와 같으면 덮어쓰고, 더 최근 월이면 추가합니다.",
        )
    with input_col2:
        manual_value = st.number_input(
            f"{series_name} 값",
            value=round(default_value, 2),
            step=0.1,
            format="%.2f",
            key=f"{series_name}_manual_value",
        )

    manual_month_ts = pd.Timestamp(manual_month).to_period("M").to_timestamp()

    info_parts = [f"현재 반영값 {current_value:.2f}"]
    if latest_auto_value is not None:
        info_parts.append(f"{latest_auto_label} {float(latest_auto_value):.2f}")
    st.caption(" / ".join(info_parts))

    result_df = base_df[base_df["date"].dt.to_period("M") != manual_month_ts.to_period("M")].copy()
    manual_row = pd.DataFrame({"date": [manual_month_ts], "value": [float(manual_value)]})
    result_df = pd.concat([result_df, manual_row], ignore_index=True)
    result_df = result_df.sort_values("date").reset_index(drop=True)

    applied_note = "덮어쓰기" if manual_month_ts <= latest_month else "추가"
    st.caption(
        f"즉시 반영 중: {manual_month_ts.strftime('%Y-%m')} = {float(manual_value):.2f} ({applied_note})"
    )
    return result_df


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
        "제조업 PMI는 신규주문, 생산, 고용, 재고, 공급망 응답을 반영하는 대표 확산지수입니다. "
        "하락은 주문 둔화와 생산 약화 가능성을 시사합니다."
    )
    impact = "제조업 PMI 약세는 경기민감주에 부담이고, 채권에는 상대적으로 우호적일 수 있습니다."
    return IndicatorSignal(
        key="pmi",
        label=SERIES_META["pmi"]["label"],
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
        buy_idea="PMI가 개선되면 제조업, 산업재, 반도체, 운송 업종 주식과 관련 ETF(XLI, SOXX, ITA 등)를 우선 검토할 수 있습니다. 반대로 50 하회가 길어지면 방어주나 중장기 국채 ETF 쪽이 더 유리할 수 있습니다.",
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
        else "80 하회로 향후 소득, 고용, 경기 기대가 약화된 상태입니다."
    )
    reason = (
        "컨퍼런스보드 소비자 기대지수는 향후 소득, 사업환경, 노동시장 전망을 반영합니다. "
        "하락은 소비 둔화 가능성과 경기 우려 확대로 해석할 수 있습니다."
    )
    impact = "기대지수 약세는 내수와 경기민감 자산에 부담이며, 방어자산 선호를 높일 수 있습니다."
    return IndicatorSignal(
        key="expectations",
        label="컨퍼런스보드 소비자 기대지수 (Consumer Expectations Index)",
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
        buy_idea="소비자 기대지수가 개선되면 소비재, 유통, 여행·레저, 온라인플랫폼 관련 주식 및 ETF(XLY, RTH 등)를 검토할 수 있습니다. 기대지수가 약하면 필수소비재(XLP)나 배당주가 상대적으로 방어적일 수 있습니다.",
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

    summary = f"전년동월비 {yoy:.1f}%로 주택 경기의 선행 흐름을 보여줍니다."
    reason = (
        "주택착공건수는 금리, 모기지 비용, 가계 구매력, 건설사 심리의 영향을 크게 받습니다. "
        "감소는 금리 부담 또는 수요 약화 신호일 수 있습니다."
    )
    impact = "주택착공 약세는 경기순환주와 리츠에 부담이 될 수 있습니다."
    return IndicatorSignal(
        key="housing",
        label=SERIES_META["housing"]["label"],
        current=current,
        previous=previous,
        change=change,
        trend_3m=trend,
        score=score,
        state=state,
        summary=summary,
        reason=reason,
        market_impact=impact,
        threshold_note="운영 기준: YoY +5 확장 / -10 이하 경고, 3개월 평균 vs 12개월 평균 보조",
        buy_idea="주택착공이 회복되면 주택건설, 건자재, 인테리어, 산업용 원자재 관련 종목과 ETF(ITB, XHB, 목재·구리 관련 ETF)를 검토할 수 있습니다. 약세면 리츠와 건설 관련주는 보수적으로 보는 편이 좋습니다.",
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
        "미국채 10년-3개월 금리차는 장기 성장 기대와 단기 정책금리의 차이를 보여줍니다. "
        "축소 또는 역전은 성장 기대 약화와 향후 금리인하 기대를 반영할 수 있습니다."
    )
    impact = "스프레드 축소와 역전은 주식보다 채권과 방어자산 쪽에 우호적인 경우가 많습니다."
    return IndicatorSignal(
        key="yield_spread",
        label=SERIES_META["yield_spread"]["label"],
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
        buy_idea="장단기 금리차가 정상화되면 경기민감주와 금융주가 유리할 수 있고, 역전이 심하면 중장기 국채 ETF(TLT, IEF)나 방어주 비중 확대가 더 유리할 수 있습니다.",
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
        "실업률은 경기 둔화가 실제 고용 약화로 이어졌는지 확인하는 대표 후행지표입니다. "
        "Sahm Rule 상승은 경기침체 신호로 자주 활용됩니다."
    )
    impact = "실업률 상승은 주식에 부담이고, 경기방어 자산에는 우호적일 수 있습니다."
    return IndicatorSignal(
        key="unemployment",
        label=SERIES_META["unemployment"]["label"],
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
        buy_idea="실업률이 안정적이면 대형주, 소비 관련주, 경기민감주 접근이 가능하지만, 실업률이 상승하면 헬스케어, 필수소비재, 유틸리티, 국채 ETF 같은 방어형 자산이 더 유리할 수 있습니다.",
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

    summary = f"소비자물가지수(CPI) YoY {current:.1f}%, 근원 CPI YoY {core_now:.1f}%입니다."
    reason = (
        "CPI는 도시 소비자가 구매하는 재화와 서비스 바스켓의 평균 가격 변화를 보여줍니다. "
        "특히 주거비와 서비스 물가가 하락 속도를 늦출 수 있습니다."
    )
    impact = "물가가 높으면 금리 인하 기대가 약해져 주식과 리츠에 부담이 될 수 있습니다."
    return IndicatorSignal(
        key="cpi",
        label="소비자물가지수(CPI) / 근원 소비자물가지수(Core CPI)",
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
        buy_idea="물가가 높고 끈적이면 금, 에너지, 원자재, 가격전가력이 높은 기업이 유리할 수 있습니다. 물가가 안정되면 성장주, 기술주, 리츠, 장기채 ETF에 우호적인 환경으로 해석할 수 있습니다.",
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
            "성장 둔화와 실업률 상승에 우호적",
            "물가 둔화가 강할수록 유리",
            "수익률곡선 역전은 방어적 수요를 자극",
        ],
        "gold": [
            "정책과 경기 불확실성에 강한 편",
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
    return " · ".join(bullet_map[asset])


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
    lead_risk = min(
        [signals["pmi"], signals["expectations"], signals["housing"], signals["yield_spread"]],
        key=lambda item: item.score,
    )
    lag_anchor = min([signals["unemployment"], signals["cpi"]], key=lambda item: item.score)
    return (
        f"현재 국면은 **{regime}**으로 판정됩니다. 선행판단에서 가장 약한 축은 **{lead_risk.label}**이며, "
        f"후행 확인에서 가장 부담되는 축은 **{lag_anchor.label}**입니다. 상대적으로 유리한 자산은 "
        f"**{top['asset']}({top['score']}점)**, 가장 보수적으로 볼 자산은 **{bottom['asset']}({bottom['score']}점)** 입니다."
    )


def make_insight_lines(regime: str, ranked_assets: pd.DataFrame, signals: Dict[str, IndicatorSignal]) -> List[str]:
    top_assets = ", ".join(ranked_assets.head(3)["asset"].tolist())
    weakest_leading = min(
        [signals["pmi"], signals["expectations"], signals["housing"], signals["yield_spread"]],
        key=lambda item: item.score,
    )
    sticky_lagging = min([signals["unemployment"], signals["cpi"]], key=lambda item: item.score)
    return [
        f"현재 매크로 국면은 **{regime}**입니다.",
        f"선행지표 가운데 가장 먼저 경계할 항목은 **{weakest_leading.label}**입니다.",
        f"후행지표 확인 축에서는 **{sticky_lagging.label}**가 아직 부담 요인입니다.",
        f"현 시점 상대 우위 자산 상위 3개는 **{top_assets}**입니다.",
    ]


def contribution_table(signals: Dict[str, IndicatorSignal]) -> pd.DataFrame:
    items = [
        ("ISM 제조업 구매관리자지수 (PMI)", signals["pmi"].score, 25),
        ("컨퍼런스보드 소비자 기대지수", signals["expectations"].score, 20),
        ("주택착공건수", signals["housing"].score, 15),
        ("미국채 10년-3개월 금리차", signals["yield_spread"].score, 25),
        ("실업률", signals["unemployment"].score, 10),
        ("소비자물가지수(CPI)", signals["cpi"].score, 5),
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


@st.cache_data(ttl=60 * 60)
def fetch_latest_ism_pmi_public() -> dict:
    try:
        res = requests.get(ISM_PMI_CURRENT_URL, headers=REQUEST_HEADERS, timeout=20)
        res.raise_for_status()
        text = res.text

        patterns = [
            r"Manufacturing PMI(?:®)?\s+at\s+([0-9]+(?:\.[0-9]+)?)%",
            r"PMI(?:®)?\s+at\s+([0-9]+(?:\.[0-9]+)?)%",
            r"manufacturing sector.*?([0-9]+(?:\.[0-9]+)?)%",
        ]

        value = None
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = float(match.group(1))
                break

        if value is None:
            return {}

        detail_link = ISM_PMI_CURRENT_URL

        return {
            "value": value,
            "label": "ISM 제조업 구매관리자지수(PMI)",
            "source": "ISM 공식 페이지",
            "link": detail_link,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return {}
    

@st.cache_data(ttl=60 * 30)
def fetch_latest_expectations_public() -> dict:
    """
    Conference Board 페이지에서 Expectations Index(CBEI)를 추출합니다.
    디버깅을 위해 매칭 문장과 일부 본문 스니펫도 함께 반환합니다.
    """
    try:
        res = requests.get(
            CONFERENCE_BOARD_CCI_URL,
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        res.raise_for_status()

        raw_text = html.unescape(res.text)
        text = re.sub(r"\s+", " ", raw_text)

        sentence_patterns = [
            r"The Expectations Index.*?(?:<|\.|;)",
            r"Expectations Index.*?(?:<|\.|;)",
        ]

        matched_sentence = ""
        for pattern in sentence_patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                matched_sentence = m.group(0)
                break

        if not matched_sentence:
            idx = text.lower().find("expectations index")
            if idx != -1:
                matched_sentence = text[max(0, idx - 40): idx + 240]

        value = None
        if matched_sentence:
            value_match = re.search(r"\bto\s+([0-9]+(?:\.[0-9]+)?)\b", matched_sentence, re.IGNORECASE)
            if value_match:
                value = float(value_match.group(1))

        if value is None:
            fallback_patterns = [
                r"The Expectations Index[^0-9]{0,80}?(?:increased|decreased|rose|fell|declined)[^0-9]{0,40}?[0-9]+(?:\.[0-9]+)?[^0-9]{0,20}?to\s+([0-9]+(?:\.[0-9]+)?)",
                r"Expectations Index[^0-9]{0,80}?(?:increased|decreased|rose|fell|declined)[^0-9]{0,40}?[0-9]+(?:\.[0-9]+)?[^0-9]{0,20}?to\s+([0-9]+(?:\.[0-9]+)?)",
                r"Expectations Index[^0-9]{0,120}?to\s+([0-9]+(?:\.[0-9]+)?)",
            ]
            for pattern in fallback_patterns:
                m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    value = float(m.group(1))
                    if not matched_sentence:
                        matched_sentence = m.group(0)
                    break

        cci_value = None
        cci_match = re.search(
            r"Consumer Confidence Index[^.]{0,160}?\bto\s+([0-9]+(?:\.[0-9]+)?)\b",
            text,
            re.IGNORECASE,
        )
        if cci_match:
            cci_value = float(cci_match.group(1))

        psi_value = None
        psi_match = re.search(
            r"Present Situation Index[^.]{0,160}?\bto\s+([0-9]+(?:\.[0-9]+)?)\b",
            text,
            re.IGNORECASE,
        )
        if psi_match:
            psi_value = float(psi_match.group(1))

        date_match = re.search(
            r"Updated:\s*[A-Za-z]+,\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            text,
            re.IGNORECASE,
        )
        release_date = date_match.group(1) if date_match else ""

        snippet = ""
        anchor = text.lower().find("expectations index")
        if anchor != -1:
            snippet = text[max(0, anchor - 80): anchor + 280]

        if value is None:
            return {
                "label": "컨퍼런스보드 소비자 기대지수 (CBEI)",
                "value": None,
                "source": "Conference Board",
                "link": CONFERENCE_BOARD_CCI_URL,
                "update_cycle": "매월 1회",
                "release_date": release_date,
                "matched_sentence": matched_sentence,
                "debug_snippet": snippet,
                "error": "Expectations Index 값을 찾지 못했습니다.",
            }

        return {
            "label": "컨퍼런스보드 소비자 기대지수 (CBEI)",
            "value": value,
            "cci_value": cci_value,
            "present_situation_value": psi_value,
            "source": "Conference Board",
            "link": CONFERENCE_BOARD_CCI_URL,
            "update_cycle": "매월 1회",
            "release_date": release_date,
            "matched_sentence": matched_sentence,
            "debug_snippet": snippet,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as exc:
        return {
            "label": "컨퍼런스보드 소비자 기대지수 (CBEI)",
            "value": None,
            "source": "Conference Board",
            "link": CONFERENCE_BOARD_CCI_URL,
            "update_cycle": "매월 1회",
            "error": f"{type(exc).__name__}: {exc}",
        }

def render_manual_update_helper() -> None:
    st.subheader("관리자 수동 업데이트 참고")

    pmi_info = fetch_latest_ism_pmi_public()
    exp_info = fetch_latest_expectations_public()

    with st.container(border=True):
        st.markdown("#### 선행지표 자동 조회 결과")
        st.caption("자동 조회가 실패하면 원문 링크를 열어 최신 수치를 확인한 뒤 아래 직접 입력란에 반영해주세요.")

        st.markdown("##### 1) ISM 제조업 PMI")
        if pmi_info:
            st.success(f"최신 조회값: {pmi_info['value']}")
            st.caption(f"출처: {pmi_info['source']} / 조회시각: {pmi_info['fetched_at']}")
            st.link_button("ISM 원문 보기", pmi_info["link"], use_container_width=True)
            st.code(f"{pd.Timestamp.today().strftime('%Y-%m-01')},{pmi_info['value']}", language="text")
        else:
            st.warning("PMI 최신 수치를 자동 조회하지 못했습니다.")
            st.link_button("ISM 원문 보기", ISM_PMI_CURRENT_URL, use_container_width=True)

        st.divider()

        st.markdown("##### 2) 소비자기대지수")
        if exp_info.get("value") is not None:
            st.success(f"최신 조회값: {float(exp_info['value']):.1f}")
            st.caption(f"출처: {exp_info['source']} / 조회시각: {exp_info.get('fetched_at', '-') }")
            st.link_button("Conference Board 원문 보기", exp_info["link"], use_container_width=True)
            st.code(f"{pd.Timestamp.today().strftime('%Y-%m-01')},{float(exp_info['value']):.1f}", language="text")
        else:
            st.warning("소비자기대지수 최신 수치를 자동 조회하지 못했습니다.")
            if exp_info.get("error"):
                st.caption(f"오류: {exp_info['error']}")
            st.link_button("Conference Board 원문 보기", CONFERENCE_BOARD_CONFIDENCE_URL, use_container_width=True)
 

def plot_series(df: pd.DataFrame, y: str, title: str, hline: Optional[float] = None) -> None:
    chart_df = df[["date", y]].dropna().copy()
    fig = px.line(
        chart_df,
        x="date",
        y=y,
        title=title,
        markers=True,
        color_discrete_sequence=[CHART_SEQUENCE[0]],
    )
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color=DASHBOARD_COLORS["navy"])
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="rgba(255,255,255,0.85)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DASHBOARD_COLORS["text"]),
        title_font=dict(size=18),
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(91,108,255,0.09)")
    st.plotly_chart(fig, use_container_width=True)


def render_signal_card(signal: IndicatorSignal, kind: str, value_format: str = "{:.1f}") -> None:
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:4px 10px;
                border-radius:999px;
                background:rgba(91,108,255,0.08);
                color:{DASHBOARD_COLORS['navy']};
                font-size:0.78rem;
                font-weight:700;
                margin-bottom:10px;
            ">{kind}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"#### {signal.label}")
        st.metric(
            label=f"{state_badge(signal.state)} {signal.state}",
            value=value_format.format(signal.current),
            delta=metric_delta(signal.change),
        )
        st.caption(signal.threshold_note)

        update_note = UPDATE_CYCLE_NOTES.get(signal.key, "")
        if update_note:
            st.caption(update_note)

        st.write(signal.summary)
        st.write(f"**왜 중요한가**: {signal.reason}")
        st.write(f"**시장 해석**: {signal.market_impact}")
        st.write(f"**매입 고려**: {signal.buy_idea}")


def render_asset_ranking(ranked_assets: pd.DataFrame) -> None:
    st.subheader("자산 랭킹")

    asset_desc_map = {
        "주식": "경기 회복·실적 개선 기대에 민감",
        "채권": "경기 둔화·금리 하락 기대에 상대 우호",
        "금": "불확실성·인플레이션 헤지 성격",
        "현금/MMF": "대기성 자금 및 변동성 방어",
        "달러": "리스크오프 구간 방어 자산",
        "리츠": "금리와 부동산 경기 민감 자산",
    }

    cards = ranked_assets.reset_index(drop=True)

    for start in range(0, len(cards), 2):
        cols = st.columns(2)
        for i in range(2):
            idx = start + i
            if idx >= len(cards):
                continue

            row = cards.iloc[idx]

            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"### #{idx + 1}. {row['asset']}")
                    st.caption(asset_desc_map.get(row["asset"], ""))
                    c1, c2 = st.columns(2)
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                display:flex;
                                justify-content:space-between;
                                align-items:center;
                                text-align:center;
                                margin-top:8px;
                                background:#F6FFDC;
                                border-radius:12px;
                                padding:10px;
                            ">
                                <div style="flex:1;">
                                    <div style="font-size:0.8rem;color:gray;">점수</div>
                                    <div style="font-size:1.4rem;font-weight:700;">{int(row['score'])}점</div>
                                </div>
                                <div style="flex:1;">
                                    <div style="font-size:0.8rem;color:gray;">등급</div>
                                    <div style="font-size:1.4rem;font-weight:700;">{row['grade']}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.caption(row["stance"])
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


def resolve_csv_input(sample_path: str, use_sample: bool, demo_loader, label: str):
    try:
        if use_sample and os.path.exists(sample_path):
            return load_sample_csv(sample_path), "프로젝트 샘플 CSV"
        return demo_loader(), f"{label} 데모 데이터"
    except Exception as exc:
        raise RuntimeError(f"{label} 데이터 로딩 실패: {exc}") from exc


def strip_html_tags(text: str) -> str:
    if text is None:
        return ""
    text = str(text)

    # CDATA 제거
    text = text.replace("<![CDATA[", "").replace("]]>", "")

    # br, p 같은 태그 개행 처리
    text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", " ", text, flags=re.IGNORECASE)

    # 나머지 태그 제거
    text = re.sub(r"<[^>]+>", "", text)

    # HTML entity 복원
    text = html.unescape(text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(ttl=60 * 10)
def fetch_bloomberg_headlines(limit: int = 10) -> pd.DataFrame:
    response = requests.get(BLOOMBERG_RSS_URL, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    items = []

    for item in root.findall("./channel/item"):
        title_raw = item.findtext("title", default="")
        link_raw = item.findtext("link", default="")
        pub_date_raw = item.findtext("pubDate", default="")

        title = strip_html_tags(title_raw)
        link = strip_html_tags(link_raw)
        pub_date = strip_html_tags(pub_date_raw)

        if not title or not link:
            continue

        translated_title = translate_text_to_korean(title)

        items.append(
            {
                "headline_en": title,
                "headline_ko": translated_title if translated_title else title,
                "link": link,
                "pub_date": pub_date,
            }
        )

        if len(items) >= limit:
            break

    return pd.DataFrame(items)


def render_headlines(headlines: pd.DataFrame) -> None:
    st.subheader("실시간 주요 뉴스 헤드라인")
    st.caption("조회 시점 기준 Bloomberg 최신 헤드라인입니다. 제목은 한글 번역으로 표시하고, 원문 기사 링크를 함께 제공합니다.")

    if headlines.empty:
        st.info("불러온 헤드라인이 없습니다.")
        return

    with st.container(border=True):
        for idx, row in headlines.iterrows():
            title_ko = strip_html_tags(row.get("headline_ko", ""))
            title_en = strip_html_tags(row.get("headline_en", ""))
            pub_date = strip_html_tags(row.get("pub_date", ""))
            link = row.get("link", "")

            st.markdown(f"**{idx + 1}. {title_ko}**")

            if title_en and title_en != title_ko:
                st.caption(title_en)

            meta_parts = []
            if pub_date:
                meta_parts.append(f"발행: {pub_date}")
            if link:
                meta_parts.append(f"[원문 링크]({link})")

            if pub_date:
                st.caption(f"발행: {pub_date}")
            if link:
                st.link_button("원문 기사 보기", link, use_container_width=False)

            if idx < len(headlines) - 1:
                st.divider()


def build_macro_summary_box(regime: str, ranked_assets: pd.DataFrame, signals: Dict[str, IndicatorSignal], risk_score: int) -> None:
    insight_lines = make_insight_lines(regime, ranked_assets, signals)
    chips = [
        f"리스크 레벨 {risk_score}/100",
        f"최상위 자산 {ranked_assets.iloc[0]['asset']}",
        f"최하위 자산 {ranked_assets.iloc[-1]['asset']}",
        f"PMI {signals['pmi'].current:.1f}",
        f"실업률 {signals['unemployment'].current:.1f}%",
        f"CPI {signals['cpi'].current:.1f}%",
    ]
    chip_html = "".join([f"<span class='insight-chip'>{html.escape(chip)}</span>" for chip in chips])
    desc = " ".join([line.replace("**", "") for line in insight_lines])
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">매크로 핵심 인사이트</div>
            <div class="hero-regime">{regime}</div>
            <div class="hero-desc">{html.escape(desc)}</div>
            <div class="insight-chip-wrap">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_dual_inflation_chart(cpi_chart: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cpi_chart["date"],
            y=cpi_chart["headline_yoy"],
            mode="lines+markers",
            name="CPI YoY",
            line=dict(width=3, color=CHART_SEQUENCE[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cpi_chart["date"],
            y=cpi_chart["core_yoy"],
            mode="lines+markers",
            name="근원 CPI YoY",
            line=dict(width=3, color=CHART_SEQUENCE[4]),
        )
    )
    fig.add_hline(y=2, line_dash="dash", line_color=DASHBOARD_COLORS["navy"])
    fig.update_layout(
        title="CPI / 근원 CPI 전년동월비 (%)",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="rgba(255,255,255,0.85)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DASHBOARD_COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(91,108,255,0.09)")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    inject_global_style()
    st.title("미국 선행·후행지표 기반 자산 배분 검토 대시보드")
    st.markdown(
        "<div class='section-note'>주요 선행 및 후행지표를 파악, 투자 흐름에 도움을 받을 수 있습니다.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("설정")
        start_date = st.date_input("조회 시작일", pd.to_datetime(DEFAULT_START_DATE))
        fred_api_key = load_api_key()

        if fred_api_key:
            st.success("FRED API Key 인식됨")
        else:
            st.warning("FRED API Key가 없습니다. .streamlit/secrets.toml 또는 환경변수에 FRED_API_KEY를 넣어주세요.")

        st.subheader("제조업 PMI 입력")
        st.caption("CSV 업로드 없이 최신 월 값을 직접 입력할 수 있습니다.")
        use_sample_pmi = st.checkbox("PMI 샘플/데모 사용", value=True)

        st.subheader("소비자 기대지수 입력")
        st.caption("CSV 업로드 없이 최신 월 값을 직접 입력할 수 있습니다.")
        use_sample_expectations = st.checkbox("Expectations 샘플/데모 사용", value=True)

        st.divider()
        with st.expander("PMI / 소비자기대지수 자동 조회 참고", expanded=False):
            render_reference_panel()

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
            pmi_sample_path,
            use_sample_pmi,
            load_demo_pmi,
            "PMI",
        )
        expectations_df, expectations_source = resolve_csv_input(
            expectations_path,
            use_sample_expectations,
            load_demo_expectations,
            "Expectations Index",
        )

        latest_pmi_auto = fetch_latest_ism_pmi_public()
        latest_exp_auto = fetch_latest_expectations_public()

        st.sidebar.divider()
        st.sidebar.subheader("즉시 반영 수동 입력")
        with st.sidebar.expander("PMI 직접 입력", expanded=True):
            pmi_df = build_manual_series_input(
                "PMI",
                pmi_df,
                latest_auto_value=latest_pmi_auto.get("value"),
                latest_auto_label="ISM 자동 조회값",
            )
            pmi_source = "사이드바 직접 입력"

        with st.sidebar.expander("소비자 기대지수 직접 입력", expanded=True):
            expectations_df = build_manual_series_input(
                "소비자 기대지수",
                expectations_df,
                latest_auto_value=latest_exp_auto.get("value"),
                latest_auto_label="Conference Board 자동 조회값",
            )
            expectations_source = "사이드바 직접 입력"

        if latest_exp_auto.get("value") is not None and not expectations_df.empty:
            input_current_exp = float(expectations_df.iloc[-1]["value"])
            auto_current_exp = float(latest_exp_auto["value"])
            if abs(input_current_exp - auto_current_exp) > 0.01:
                st.warning(
                    f"직접 입력된 소비자 기대지수 값({input_current_exp:.1f})과 자동조회 값({auto_current_exp:.1f})이 다릅니다. "
                    "의도한 값이면 그대로 사용하셔도 됩니다."
                )

        if latest_pmi_auto.get("value") is not None and not pmi_df.empty:
            input_current_pmi = float(pmi_df.iloc[-1]["value"])
            auto_current_pmi = float(latest_pmi_auto["value"])
            if abs(input_current_pmi - auto_current_pmi) > 0.01:
                st.info(
                    f"직접 입력된 PMI 값({input_current_pmi:.1f})이 자동조회 값({auto_current_pmi:.1f})과 다릅니다. "
                    "수정값이 즉시 전체 계산에 반영됩니다."
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
    risk_score = int(clip_score(50 + (-lead_score * 18) + (max(0, -lag_score) * 12)))

    build_macro_summary_box(regime, ranked_assets, signals, risk_score)

    top_left, top_mid, top_right = st.columns([1.0, 1.0, 2.4])
    with top_left:
        st.metric("현재 경기 국면", regime)
    with top_mid:
        st.metric("리스크 레벨", f"{risk_score}/100")
    with top_right:
        st.info(narrative_for_dashboard(regime, ranked_assets, signals))

    news_col, ranking_col = st.columns([1.15, 1.35])
    with news_col:
        try:
            headlines = fetch_bloomberg_headlines(10)
            render_headlines(headlines)
        except Exception as exc:
            st.warning(f"Bloomberg 헤드라인을 불러오지 못했습니다: {exc}")
    with ranking_col:
        render_asset_ranking(ranked_assets)

    st.divider()
    report_col, quick_col = st.columns([1.7, 1.1])
    with report_col:
        st.subheader("자동 해석 리포트")
        st.markdown(
            textwrap.dedent(
                f"""
                - **ISM 제조업 구매관리자지수 (PMI)**: {signals['pmi'].summary}
                - **컨퍼런스보드 소비자 기대지수**: {signals['expectations'].summary}
                - **주택착공건수**: {signals['housing'].summary}
                - **미국채 10년-3개월 금리차**: {signals['yield_spread'].summary}
                - **실업률**: {signals['unemployment'].summary}
                - **소비자물가지수(CPI)**: {signals['cpi'].summary}
                """
            )
        )
        st.markdown("#### 투자 방향 해석")
        st.write(
            f"상위 자산군은 **{', '.join(ranked_assets.head(3)['asset'].tolist())}** 입니다. "
            f"현재 국면이 **{regime}**로 판정된 만큼, 단일 자산 편중보다는 상위 랭킹 자산 중심의 분산과 하위 랭킹 자산 노출 관리가 핵심입니다."
        )
        st.caption(f"PMI 소스: {pmi_source} / Expectations Index 소스: {expectations_source}")
    with quick_col:
        st.subheader("핵심 체크포인트")
        for line in make_insight_lines(regime, ranked_assets, signals):
            st.markdown(f"- {line}")

    st.divider()
    st.subheader("선행지표")
    st.caption("※ ISM 제조업 PMI와 컨퍼런스보드 소비자 기대지수는 사이드바 직접 입력값이 즉시 반영됩니다.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_signal_card(signals["pmi"], "선행지표", "{:.1f}")
    with c2:
        render_signal_card(signals["expectations"], "선행지표", "{:.1f}")
    with c3:
        render_signal_card(signals["housing"], "선행지표", "{:.0f}")
    with c4:
        render_signal_card(signals["yield_spread"], "선행지표", "{:.2f}")

    st.subheader("후행지표")
    c5, c6 = st.columns(2)
    with c5:
        render_signal_card(signals["unemployment"], "후행지표", "{:.1f}")
    with c6:
        render_signal_card(signals["cpi"], "후행지표", "{:.1f}")

    st.divider()
    tabs = st.tabs(["전체 요약", "선행지표 차트", "후행지표 차트", "자산 점수 근거", "데이터 다운로드"])

    with tabs[0]:
        st.dataframe(contribution_table(signals), use_container_width=True, hide_index=True)
        st.dataframe(ranked_assets[["asset", "score", "grade", "stance"]], use_container_width=True, hide_index=True)

    with tabs[1]:
        plot_series(combined, "pmi", "ISM 제조업 구매관리자지수 (PMI)", hline=50)
        plot_series(combined, "expectations", "컨퍼런스보드 소비자 기대지수", hline=80)
        housing_chart = combined[["date", "housing"]].dropna().copy()
        housing_chart["housing_yoy"] = housing_chart["housing"].pct_change(12) * 100
        plot_series(housing_chart, "housing_yoy", "주택착공건수 전년동월비 (%)", hline=0)
        plot_series(combined, "yield_spread", "미국채 10년-3개월 금리차", hline=0)

    with tabs[2]:
        plot_series(combined, "unemployment", "실업률", hline=None)
        cpi_chart = combined[["date", "cpi", "core_cpi"]].dropna().copy()
        cpi_chart["headline_yoy"] = cpi_chart["cpi"].pct_change(12) * 100
        cpi_chart["core_yoy"] = cpi_chart["core_cpi"].pct_change(12) * 100
        plot_dual_inflation_chart(cpi_chart)

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
        latest_exp_debug = fetch_latest_expectations_public()
        st.write({'expectations_auto_debug': latest_exp_debug})


if __name__ == "__main__":
    main()
