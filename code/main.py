
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# =====================================
# 1. 페이지 설정
# =====================================
st.set_page_config(
    page_title="지역별 의료환경 데이터 분석",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 지역별 의료환경과 건강수준 분석")
st.markdown("""
**탐구 주제:** 지역별 인구 1만 명당 의사 수와 주관적 건강수준의 관계

2021년 전국 17개 시도 데이터를 이용한 통계 분석 페이지입니다.
""")

st.divider()

# =====================================
# 2. 데이터 직접 입력
# =====================================
data = {
    "지역": [
        "서울", "부산", "대구", "인천", "광주",
        "대전", "울산", "세종", "경기", "강원",
        "충북", "충남", "전북", "전남", "경북",
        "경남", "제주"
    ],

    "연도": [2021] * 17,

    "인구 1만 명당 의사 수": [
        47, 35, 36, 26, 37, 37, 24, 19, 25,
        27, 24, 24, 31, 26, 21, 25, 26
    ],

    "주관적 건강수준 인지율(%)": [
        52.9, 46.9, 47.4, 45.6, 48.6, 55.2,
        48.6, 48.9, 48.6, 48.0, 47.8, 48.4,
        43.9, 48.7, 47.4, 45.6, 52.5
    ],

    "총인구": [
        9509000, 3350000, 2385000, 2948000,
        1442000, 1452000, 1122000, 372000,
        13565000, 1538000, 1597000, 2119000,
        1787000, 1833000, 2627000, 3314000,
        677000
    ],

    "고령인구 비율(%)": [
        16.8, 20.4, 17.6, 14.9, 14.9, 15.1,
        13.7, 9.9, 14.1, 21.6, 18.7, 19.6,
        22.2, 24.5, 22.6, 18.4, 16.5
    ],

    "인구 1만 명당 의료기관 수": [
        None
    ] * 17
}

df = pd.DataFrame(data)

# =====================================
# 3. 기본 데이터 정보
# =====================================
st.header("📊 1. 데이터 기본 정보")

c1, c2, c3 = st.columns(3)

c1.metric("분석 지역", "17개")
c2.metric("분석 연도", "2021년")
c3.metric("분석 변수", "5개 수치형 변수")

st.subheader("지역별 전체 데이터")
st.dataframe(df, use_container_width=True)

st.caption(
    "주의: 아래 수치는 이전에 작성한 CSV의 값으로, "
    "KOSIS 원자료와 대조 및 검증이 완료되지 않았습니다. "
    "탐구 보고서에 사용하기 전에 원자료 확인이 필요합니다."
)

st.divider()

# =====================================
# 4. 지역별 의사 수 비교
# =====================================
st.header("📍 2. 지역별 의사 수 비교")

fig1 = px.bar(
    df,
    x="지역",
    y="인구 1만 명당 의사 수",
    color="인구 1만 명당 의사 수",
    title="지역별 인구 1만 명당 의사 수",
    text="인구 1만 명당 의사 수",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig1, width="stretch")

# =====================================
# 5. 건강수준 비교
# =====================================
st.header("❤️ 3. 지역별 주관적 건강수준")

fig2 = px.bar(
    df,
    x="지역",
    y="주관적 건강수준 인지율(%)",
    color="주관적 건강수준 인지율(%)",
    title="지역별 주관적 건강수준 인지율",
    text="주관적 건강수준 인지율(%)",
    color_continuous_scale="Greens"
)

st.plotly_chart(fig2, width="stretch")

st.divider()

# =====================================
# 6. 의사 수와 건강수준 관계
# =====================================
st.header("🔎 4. 의사 수와 건강수준의 관계")

corr1 = df[
    "인구 1만 명당 의사 수"
].corr(
    df["주관적 건강수준 인지율(%)"]
)

st.metric(
    "의사 수와 건강수준의 상관계수",
    f"{corr1:.3f}"
)

fig3 = px.scatter(
    df,
    x="인구 1만 명당 의사 수",
    y="주관적 건강수준 인지율(%)",
    text="지역",
    hover_name="지역",
    title="의사 수와 주관적 건강수준의 관계",
    trendline=None
)

# 추세선 추가
x = df["인구 1만 명당 의사 수"]
y = df["주관적 건강수준 인지율(%)"]

slope, intercept = np.polyfit(x, y, 1)

x_line = np.linspace(x.min(), x.max(), 100)

fig3.add_scatter(
    x=x_line,
    y=slope * x_line + intercept,
    mode="lines",
    name="선형 추세선"
)

st.plotly_chart(fig3, width="stretch")

st.divider()

# =====================================
# 7. 추가 변수 분석
# =====================================
st.header("📈 5. 추가 변수 분석")

numeric_cols = [
    "인구 1만 명당 의사 수",
    "주관적 건강수준 인지율(%)",
    "총인구",
    "고령인구 비율(%)"
]

col1, col2 = st.columns(2)

with col1:
    x_var = st.selectbox(
        "X축 변수 선택",
        numeric_cols,
        index=0
    )

with col2:
    y_var = st.selectbox(
        "Y축 변수 선택",
        numeric_cols,
        index=1
    )

if x_var != y_var:

    corr = df[x_var].corr(df[y_var])

    st.metric(
        "선택한 변수의 상관계수",
        f"{corr:.3f}"
    )

    fig4 = px.scatter(
        df,
        x=x_var,
        y=y_var,
        text="지역",
        hover_name="지역",
        title=f"{x_var}와 {y_var}의 관계"
    )

    st.plotly_chart(fig4, width="stretch")

else:
    st.warning("서로 다른 두 변수를 선택해 주세요.")

st.divider()

# =====================================
# 8. 전체 상관관계 분석
# =====================================
st.header("🧮 6. 전체 변수 상관관계 분석")

corr_matrix = df[numeric_cols].corr()

fig5 = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="변수 간 상관관계 히트맵"
)

st.plotly_chart(fig5, width="stretch")

st.divider()

# =====================================
# 9. CSV 다운로드
# =====================================
st.header("📥 7. 데이터 다운로드")

csv = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    "분석 데이터 CSV 다운로드",
    data=csv,
    file_name="지역별_의료환경_분석.csv",
    mime="text/csv"
)

st.success("데이터 분석 페이지 실행이 완료되었습니다!")
