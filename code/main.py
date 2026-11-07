
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

st.markdown("""
KOSIS 통계를 활용하여 지역별 의료환경과 주관적 건강수준의 관계를 분석합니다.
""")

st.divider()


# ==========================================
# 2. 데이터 입력
# ==========================================

# 의사 수:
# 2021년 인구 1천 명당 의사 수를 10배하여
# 인구 1만 명당 의사 수로 환산한 값
#
# 출처: 통계청, 한국의 SDG 이행보고서 2024
#
# 건강수준:
# 이전 데이터 초안의 값이며 17개 지역 전체의
# KOSIS 원자료 대조가 아직 완료되지 않음.

data = {
    "지역": [
        "서울", "부산", "대구", "인천", "광주",
        "대전", "울산", "세종", "경기", "강원",
        "충북", "충남", "전북", "전남", "경북",
        "경남", "제주"
    ],

    "연도": [2021] * 17,

    "인구 1만 명당 의사 수": [
        39, 30, 31, 21, 30,
        30, 20, 16, 20, 22,
        19, 19, 26, 23, 18,
        21, 21
    ],

    "주관적 건강수준 인지율(%)": [
        52.9, 46.9, 47.4, 45.6, 48.6,
        55.2, 48.6, 48.9, 48.6, 48.0,
        47.8, 48.4, 43.9, 48.7, 47.4,
        45.6, 52.5
    ]
}

df = pd.DataFrame(data)

doctor = "인구 1만 명당 의사 수"
health = "주관적 건강수준 인지율(%)"


# ==========================================
# 3. 데이터 출처 안내
# ==========================================

with st.expander("데이터 출처 및 검증 상태 확인"):

    st.markdown("""
    **의사 수**

    - 기준 연도: 2021년
    - 원자료 단위: 인구 1천 명당 의사 수
    - 코드 단위: 인구 1만 명당 의사 수
    - 환산 방법: 원자료 × 10
    - 출처: 통계청 「한국의 SDG 이행보고서 2024」
    - 원자료: 국민건강보험공단 지역별 의료이용통계

    **주관적 건강수준 인지율**

    - 기준 연도: 2021년으로 설정
    - 기존 탐구용 데이터 초안에서 가져온 값
    - 17개 지역 전체의 공식 원자료 대조가 필요함

    따라서 현재 분석 결과는 최종 검증된 통계 분석이 아니라
    데이터 구조와 분석 기능을 확인하기 위한 임시 결과입니다.
    """)

st.warning(
    "주의: 주관적 건강수준 인지율은 17개 지역 전체의 원자료 검증 전입니다. "
    "최종 탐구 결과로 제출하기 전에 공식 수치로 교체해야 합니다."
)


# ==========================================
# 4. 상단 메뉴
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
    1. 지역별 인구 1만 명당 의사 수에는 어떤 차이가 있을까?
    2. 의사 수와 주관적 건강수준 사이에는 상관관계가 있을까?
    3. 두 변수 사이의 관계를 그래프로 표현하면 어떤 특징이 나타날까?
    4. 의사 수를 이용하여 주관적 건강수준을 예측할 수 있을까?
    """)

    st.divider()

    st.subheader("탐구 가설")

    st.write("""
    인구 1만 명당 의사 수가 많은 지역일수록
    의료서비스를 이용할 수 있는 기회가 많아져
    주관적 건강수준 인지율도 높게 나타날 것이다.
    """)

    st.divider()

    st.subheader("탐구 목표")

    st.write("""
    전국 17개 시도의 의료환경과 건강수준 자료를 활용하여
    두 변수 사이의 관계를 분석한다.

    이후 선형회귀 모델을 이용하여 건강수준의 예측 가능성을 확인한다.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("분석 지역", "17개")
    c2.metric("기준 연도", "2021년")
    c3.metric("핵심 변수", "2개")


# ==========================================
# TAB 2. 데이터 분석
# ==========================================

with tab2:

    st.header("2. 데이터 분석")

    st.subheader("① 원본 데이터")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("② 지역별 의사 수 비교")

    fig1 = px.bar(
        df,
        x="지역",
        y=doctor,
        color=doctor,
        text=doctor,
        title="지역별 인구 1만 명당 의사 수",
        color_continuous_scale="Blues"
    )

    fig1.update_layout(
        xaxis_title="지역",
        yaxis_title="의사 수(명)"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    st.subheader("③ 지역별 주관적 건강수준 비교")

    fig2 = px.bar(
        df,
        x="지역",
        y=health,
        color=health,
        text=health,
        title="지역별 주관적 건강수준 인지율",
        color_continuous_scale="Greens"
    )

    fig2.update_layout(
        xaxis_title="지역",
        yaxis_title="인지율(%)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("④ 두 변수의 관계")

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
        hover_name="지역",
        title="의사 수와 주관적 건강수준의 관계"
    )

    fig3.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    if corr > 0:
        st.info("두 변수 사이에 양의 상관관계가 나타납니다.")
    elif corr < 0:
        st.info("두 변수 사이에 음의 상관관계가 나타납니다.")
    else:
        st.info("두 변수 사이에 선형 상관관계가 거의 없습니다.")

    st.caption(
        "상관관계가 나타나더라도 의사 수가 건강수준을 직접적으로 "
        "높였다고 단정할 수는 없습니다."
    )

    st.divider()

    st.subheader("⑤ 상관관계 히트맵")

    corr_matrix = df[[doctor, health]].corr()

    fig4 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="두 변수의 상관관계 히트맵"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# ==========================================
# TAB 3. 기준선 모델
# ==========================================

with tab3:

    st.header("3. 기준선 모델")

    st.subheader("기준선 모델이란?")

    st.write("""
    기준선 모델은 모든 지역의 건강수준을
    전체 지역의 평균값으로 예측하는 간단한 모델이다.

    이 모델을 기준으로 선형회귀 모델의 예측 성능을 비교한다.
    """)

    baseline = df[health].mean()

    st.metric(
        "전체 지역 평균 건강수준",
        f"{baseline:.2f}%"
    )

    st.divider()

    st.subheader("실제값과 기준선 예측값 비교")

    baseline_df = df[["지역", health]].copy()

    baseline_df["기준선 예측값"] = baseline

    fig5 = go.Figure()

    fig5.add_trace(
        go.Bar(
            x=baseline_df["지역"],
            y=baseline_df[health],
            name="실제값"
        )
    )

    fig5.add_trace(
        go.Scatter(
            x=baseline_df["지역"],
            y=baseline_df["기준선 예측값"],
            mode="lines",
            name="기준선 예측값",
            line=dict(dash="dash")
        )
    )

    fig5.update_layout(
        title="실제 건강수준과 기준선 예측값",
        xaxis_title="지역",
        yaxis_title="건강수준 인지율(%)"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.divider()

    st.subheader("기준선 모델의 역할")

    st.write("""
    기준선 모델은 복잡한 계산 없이 평균값만으로 예측한다.

    선형회귀 모델이 이보다 더 정확하게 예측하는지 확인하면
    의사 수라는 변수를 추가하는 것이 예측에 도움이 되는지
    탐색할 수 있다.
    """)


# ==========================================
# TAB 4. 모델 평가
# ==========================================

with tab4:

    st.header("4. 모델 평가")

    st.subheader("① 선형회귀 모델")

    st.write("""
    독립변수(X): 인구 1만 명당 의사 수

    종속변수(Y): 주관적 건강수준 인지율

    의사 수를 이용하여 건강수준을 예측하는
    단순 선형회귀 모델을 구성한다.
    """)

    X = df[[doctor]]
    y = df[health]

    model = LinearRegression()

    loo = LeaveOneOut()

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

    baseline_predictions = []

    for train_idx, test_idx in loo.split(X):

        train_mean = y.iloc[train_idx].mean()

        baseline_predictions.append(train_mean)

    # 평가 지표 계산

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(
        y,
        predictions
    )

    baseline_mae = mean_absolute_error(
        y,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y,
            baseline_predictions
        )
    )

    st.divider()

    st.subheader("② 모델 성능 평가")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("R²", f"{r2:.3f}")

    st.markdown("""
    - **MAE:** 실제값과 예측값의 평균적인 차이
    - **RMSE:** 예측 오차를 제곱하여 계산한 평균적인 크기
    - **R²:** 모델이 데이터의 변동을 얼마나 설명하는지 나타내는 지표
    """)

    st.divider()

    st.subheader("③ 기준선 모델과 선형회귀 비교")

    comparison = pd.DataFrame({
        "모델": [
            "기준선 모델",
            "선형회귀 모델"
        ],

        "MAE": [
            baseline_mae,
            mae
        ],

        "RMSE": [
            baseline_rmse,
            rmse
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    fig6 = px.bar(
        comparison,
        x="모델",
        y=["MAE", "RMSE"],
        barmode="group",
        title="모델별 예측 오차 비교"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    st.divider()

    st.subheader("④ 실제값과 예측값 비교")

    result = pd.DataFrame({
        "지역": df["지역"],
        "실제값": y,
        "예측값": predictions
    })

    fig7 = px.scatter(
        result,
        x="실제값",
        y="예측값",
        text="지역",
        hover_name="지역",
        title="실제 건강수준과 예측 건강수준"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("⑤ 최종 평가")

    if mae < baseline_mae:

        st.success(
            "선형회귀 모델이 기준선 모델보다 "
            "낮은 MAE를 기록했습니다."
        )

    else:

        st.warning(
            "선형회귀 모델이 기준선 모델보다 "
            "낮은 MAE를 기록하지 못했습니다."
        )

    st.write("""
    분석 대상이 전국 17개 시도로 제한되어 있어
    표본 수가 적다는 한계가 있다.

    따라서 모델의 예측 성능을 일반화하기 어렵고,
    결과를 탐색적인 분석으로 해석해야 한다.
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
    file_name="지역별_의료환경_건강수준.csv",
    mime="text/csv"
)

st.caption(
    "의사 수 출처: 통계청 「한국의 SDG 이행보고서 2024」 "
    "및 국민건강보험공단 지역별 의료이용통계"
)
