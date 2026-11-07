
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="지역별 의료환경과 건강수준 분석",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 지역별 의료환경과 건강수준 분석")

st.write(
    "지역별 인구 1만 명당 의사 수와 "
    "주관적 건강수준 인지율의 관계를 분석합니다."
)


# ==========================================
# 2. 데이터 입력
# ==========================================

# 주관적 건강수준 인지율 공식 확인값
# 출처: 질병관리청 지역사회건강조사
# 2021년 수도권 광역지표 분석 보고서

data = {
    "지역": [
        "서울",
        "인천",
        "경기"
    ],

    "연도": [
        2021,
        2021,
        2021
    ],

    # 기존 코드의 의사 수 값을 사용한 임시 입력값
    # 제출 전 의사 수 원자료도 별도 확인할 것
    "인구 1만 명당 의사 수": [
        39,
        21,
        20
    ],

    # 공식 보고서에서 확인한 수치
    "주관적 건강수준 인지율(%)": [
        52.9,
        45.6,
        48.6
    ]
}

df = pd.DataFrame(data)

doctor = "인구 1만 명당 의사 수"
health = "주관적 건강수준 인지율(%)"


# ==========================================
# 3. 메뉴
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "1. 질문",
    "2. 데이터 분석",
    "3. 기준선",
    "4. 모델 평가"
])


# ==========================================
# TAB 1. 질문
# ==========================================

with tab1:

    st.header("1. 탐구 질문")

    st.subheader("핵심 탐구 질문")

    st.info(
        "지역별 인구 1만 명당 의사 수가 많을수록 "
        "주관적 건강수준 인지율도 높을까?"
    )

    st.subheader("세부 탐구 질문")

    st.markdown("""
    1. 서울, 인천, 경기의 의사 수에는 어떤 차이가 있을까?
    2. 의사 수와 주관적 건강수준 사이에는 관계가 있을까?
    3. 의사 수를 활용하여 건강수준을 예측할 수 있을까?
    """)

    st.divider()

    st.subheader("탐구 가설")

    st.write("""
    인구 1만 명당 의사 수가 많은 지역일수록
    의료서비스를 이용할 수 있는 기회가 많아져
    주관적 건강수준 인지율도 높게 나타날 것이다.
    """)

    st.warning(
        "현재 공식 건강수준 수치를 확인한 지역은 3개입니다. "
        "전국 17개 시도로 일반화할 수 없습니다."
    )


# ==========================================
# TAB 2. 데이터 분석
# ==========================================

with tab2:

    st.header("2. 데이터 분석")

    st.subheader("① 데이터 표")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "건강수준 인지율 출처: 질병관리청 지역사회건강조사"
    )

    st.divider()

    st.subheader("② 지역별 의사 수")

    fig1 = px.bar(
        df,
        x="지역",
        y=doctor,
        text=doctor,
        title="지역별 인구 1만 명당 의사 수"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 지역별 주관적 건강수준")

    fig2 = px.bar(
        df,
        x="지역",
        y=health,
        text=health,
        title="지역별 주관적 건강수준 인지율"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    st.subheader("④ 의사 수와 건강수준의 관계")

    corr = df[doctor].corr(df[health])

    st.metric(
        "피어슨 상관계수",
        f"{corr:.3f}"
    )

    fig3 = px.scatter(
        df,
        x=doctor,
        y=health,
        text="지역",
        title="의사 수와 건강수준의 관계"
    )

    fig3.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.warning(
        "표본이 3개 지역뿐이므로 상관계수는 참고용으로만 해석해야 합니다."
    )


# ==========================================
# TAB 3. 기준선 모델
# ==========================================

with tab3:

    st.header("3. 기준선 모델")

    st.write("""
    기준선 모델은 모든 지역의 건강수준을
    전체 지역의 평균값으로 예측하는 모델입니다.
    """)

    baseline = df[health].mean()

    st.metric(
        "세 지역의 평균 건강수준",
        f"{baseline:.2f}%"
    )

    baseline_df = df[["지역", health]].copy()

    baseline_df["기준선 예측값"] = baseline

    fig4 = go.Figure()

    fig4.add_trace(
        go.Bar(
            x=baseline_df["지역"],
            y=baseline_df[health],
            name="실제값"
        )
    )

    fig4.add_trace(
        go.Scatter(
            x=baseline_df["지역"],
            y=baseline_df["기준선 예측값"],
            mode="lines",
            name="기준선 예측값"
        )
    )

    fig4.update_layout(
        title="실제 건강수준과 기준선 예측값 비교",
        xaxis_title="지역",
        yaxis_title="건강수준 인지율(%)"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# ==========================================
# TAB 4. 모델 평가
# ==========================================

with tab4:

    st.header("4. 모델 평가")

    st.subheader("선형회귀 모델")

    st.write("""
    독립변수(X): 인구 1만 명당 의사 수

    종속변수(Y): 주관적 건강수준 인지율
    """)

    st.error(
        "현재 데이터는 3개 지역뿐입니다. "
        "이 표본으로 교차검증이나 예측 성능을 평가하면 "
        "결과가 매우 불안정하므로 최종 모델 성능으로 제시하지 않습니다."
    )

    st.subheader("분석 데이터")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("향후 모델 평가 조건")

    st.markdown("""
    전국 17개 시도의 공식 자료를 모두 확보한 후 다음 지표를 계산합니다.

    - MAE: 평균 절대 오차
    - RMSE: 평균 제곱근 오차
    - R²: 결정계수
    - 기준선 모델과 선형회귀 모델의 성능 비교
    """)


# ==========================================
# 5. 데이터 다운로드
# ==========================================

st.divider()

st.header("📥 데이터 다운로드")

csv = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="분석 데이터 CSV 다운로드",
    data=csv,
    file_name="공식건강수준_확인자료.csv",
    mime="text/csv"
)

st.caption(
    "주관적 건강수준 자료: 질병관리청 지역사회건강조사, 2021년"
)
