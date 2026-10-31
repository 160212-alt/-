
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ==================================
# 1. 페이지 기본 설정
# ==================================

st.set_page_config(
    page_title="KOSIS 의료·건강 데이터 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 지역별 의료 접근성과 건강수준 분석")
st.markdown("""
**KOSIS 국가통계포털 데이터를 활용한 데이터 과학 탐구**

지역별 인구 1만 명당 의사 수와 주관적 건강수준의 관계를 분석합니다.
""")

st.divider()

# ==================================
# 2. 데이터 불러오기
# ==================================

st.sidebar.header("📁 데이터 설정")

uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 업로드",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
elif os.path.exists("data.csv"):
    df = pd.read_csv("data.csv", encoding="utf-8-sig")
else:
    st.warning("data.csv 파일을 GitHub에 업로드하거나 CSV 파일을 직접 업로드해주세요.")
    st.stop()

# ==================================
# 3. 컬럼명 통일
# ==================================

column_aliases = {
    "지역": ["지역", "시군구", "시도", "행정구역"],
    "연도": ["연도", "기준연도", "년도"],
    "의사수": [
        "인구 1만 명당 의사 수",
        "인구1만명당의사수",
        "만명당 의사 수",
        "만명당의사수"
    ],
    "건강수준": [
        "주관적 건강수준 인지율(%)",
        "주관적 건강수준",
        "건강수준 인지율",
        "주관적 건강수준 인지율"
    ],
    "인구": ["총인구", "인구"],
    "고령인구비율": ["고령인구 비율(%)", "고령인구비율"],
    "의료기관수": [
        "인구 1만 명당 의료기관 수",
        "만명당 의료기관 수"
    ]
}

rename_dict = {}

for standard_name, aliases in column_aliases.items():
    for alias in aliases:
        if alias in df.columns:
            rename_dict[alias] = standard_name
            break

df = df.rename(columns=rename_dict)

# ==================================
# 4. 데이터 검증 및 전처리
# ==================================

required_columns = ["지역", "의사수", "건강수준"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"필수 데이터 열이 없습니다: {missing_columns}\n\n"
        f"현재 CSV 열 이름: {list(df.columns)}"
    )
    st.stop()

numeric_columns = [
    "연도", "의사수", "건강수준",
    "인구", "고령인구비율", "의료기관수"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.replace("%", ""),
            errors="coerce"
        )

df["지역"] = df["지역"].astype(str).str.strip()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["지역", "의사수", "건강수준"])

df = df[
    (df["의사수"] >= 0) &
    (df["건강수준"].between(0, 100))
]

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

# ==================================
# 5. 사이드바 필터
# ==================================

st.sidebar.divider()
st.sidebar.header("🔍 분석 조건")

if "연도" in df.columns and df["연도"].notna().any():
    years = sorted(df["연도"].dropna().unique())

    selected_years = st.sidebar.multiselect(
        "분석 연도",
        years,
        default=years
    )

    if selected_years:
        filtered_df = df[df["연도"].isin(selected_years)].copy()
    else:
        filtered_df = df.iloc[0:0].copy()
else:
    filtered_df = df.copy()

regions = sorted(filtered_df["지역"].unique())

selected_regions = st.sidebar.multiselect(
    "분석 지역",
    regions,
    default=regions
)

filtered_df = filtered_df[
    filtered_df["지역"].isin(selected_regions)
]

if filtered_df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ==================================
# 6. 핵심 통계
# ==================================

st.subheader("📌 데이터 핵심 지표")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "분석 데이터 수",
    f"{len(filtered_df):,}개"
)

col2.metric(
    "평균 의사 수",
    f"{filtered_df['의사수'].mean():.2f}명"
)

col3.metric(
    "평균 건강수준",
    f"{filtered_df['건강수준'].mean():.2f}%"
)

correlation = filtered_df[
    ["의사수", "건강수준"]
].corr().iloc[0, 1]

col4.metric(
    "피어슨 상관계수",
    f"{correlation:.3f}" if pd.notna(correlation) else "계산 불가"
)

st.divider()

# ==================================
# 7. 지역별 의사 수 분석
# ==================================

st.subheader("🏥 지역별 인구 1만 명당 의사 수")

doctor_df = (
    filtered_df.groupby("지역", as_index=False)["의사수"]
    .mean()
    .sort_values("의사수", ascending=False)
)

fig1 = px.bar(
    doctor_df,
    x="지역",
    y="의사수",
    color="의사수",
    color_continuous_scale="Blues",
    text_auto=".2f",
    title="지역별 평균 인구 1만 명당 의사 수"
)

fig1.update_layout(
    xaxis_title="지역",
    yaxis_title="의사 수 (명)",
    coloraxis_showscale=False
)

st.plotly_chart(fig1, use_container_width=True)

# ==================================
# 8. 주관적 건강수준 분석
# ==================================

st.subheader("💚 지역별 주관적 건강수준")

health_df = (
    filtered_df.groupby("지역", as_index=False)["건강수준"]
    .mean()
    .sort_values("건강수준", ascending=False)
)

fig2 = px.bar(
    health_df,
    x="지역",
    y="건강수준",
    color="건강수준",
    color_continuous_scale="Greens",
    text_auto=".2f",
    title="지역별 평균 주관적 건강수준 인지율"
)

fig2.update_layout(
    xaxis_title="지역",
    yaxis_title="건강수준 인지율 (%)",
    coloraxis_showscale=False
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ==================================
# 9. 상관관계 분석
# ==================================

st.subheader("📈 의사 수와 주관적 건강수준의 관계")

fig3 = px.scatter(
    filtered_df,
    x="의사수",
    y="건강수준",
    hover_name="지역",
    color="지역",
    size="인구" if "인구" in filtered_df.columns else None,
    trendline="ols" if len(filtered_df) >= 2 else None,
    title="인구 1만 명당 의사 수와 건강수준의 상관관계",
    labels={
        "의사수": "인구 1만 명당 의사 수",
        "건강수준": "주관적 건강수준 인지율 (%)"
    }
)

st.plotly_chart(fig3, use_container_width=True)

# ==================================
# 10. 상관관계 해석
# ==================================

st.subheader("🔎 상관관계 분석 결과")

if pd.notna(correlation):

    if correlation >= 0.7:
        interpretation = "강한 양의 상관관계"
    elif correlation >= 0.3:
        interpretation = "중간 정도의 양의 상관관계"
    elif correlation > 0:
        interpretation = "약한 양의 상관관계"
    elif correlation <= -0.7:
        interpretation = "강한 음의 상관관계"
    elif correlation <= -0.3:
        interpretation = "중간 정도의 음의 상관관계"
    elif correlation < 0:
        interpretation = "약한 음의 상관관계"
    else:
        interpretation = "선형 상관관계가 거의 없음"

    st.info(
        f"현재 선택한 데이터에서 의사 수와 건강수준은 "
        f"'{interpretation}'를 보입니다. "
        f"(상관계수: {correlation:.3f})"
    )

    st.caption(
        "주의: 상관관계가 있다고 해서 의사 수가 건강수준의 "
        "직접적인 원인이라고 단정할 수는 없습니다."
    )
else:
    st.warning("상관계수를 계산하기에 데이터가 부족합니다.")

# ==================================
# 11. 추가 변수 분석
# ==================================

st.divider()
st.subheader("📊 추가 변수 분석")

optional_columns = [
    col for col in ["고령인구비율", "의료기관수"]
    if col in filtered_df.columns
]

if optional_columns:
    selected_variable = st.selectbox(
        "분석할 추가 변수 선택",
        optional_columns
    )

    additional_df = filtered_df.dropna(
        subset=[selected_variable, "건강수준"]
    )

    if len(additional_df) >= 2:
        fig4 = px.scatter(
            additional_df,
            x=selected_variable,
            y="건강수준",
            color="지역",
            hover_name="지역",
            trendline="ols",
            title=f"{selected_variable}과 주관적 건강수준의 관계"
        )

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("추가 분석을 위한 데이터가 부족합니다.")
else:
    st.info("추가 변수 데이터가 없습니다.")

# ==================================
# 12. 원본 데이터 확인
# ==================================

st.divider()
st.subheader("📋 원본 데이터 확인")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="📥 분석 데이터 CSV 다운로드",
    data=csv_data,
    file_name="KOSIS_건강수준_분석.csv",
    mime="text/csv"
)

st.caption(
    "자료 출처: KOSIS 국가통계포털 | "
    "분석 결과는 업로드한 원자료의 범위와 품질에 따라 달라집니다."
)

