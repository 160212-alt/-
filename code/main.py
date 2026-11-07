
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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

    # 기존 코드의 임시 입력값
    # 공식 의사 수 원자료 확인 필요
    "인구 1만 명당 의사 수": [
        39,
        21,
        20
    ],

    # 주관적 건강수준 인지율
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
        "현재 건강수준 인지율을 확인한 지역은 3개입니다. "
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
    기준선 모델은 복잡한 예측 방법을 사용하지 않고
    학습 데이터의 평균값만으로 예측하는 가장 단순한 모델입니다.
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
            mode="lines+markers",
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

    st.subheader("기준선 모델의 특징")

    st.markdown("""
    - 모든 지역에 동일한 평균값을 예측합니다.
    - 지역별 의사 수 차이를 반영하지 않습니다.
    - 선형회귀 모델의 성능을 평가할 때 비교 기준으로 사용합니다.
    """)


# ==========================================
# TAB 4. 모델 평가
# ==========================================

with tab4:

    st.header("4. 모델 평가")

    st.subheader("① 평가 방법")

    st.write("""
    독립변수(X): 인구 1만 명당 의사 수

    종속변수(Y): 주관적 건강수준 인지율
    """)

    st.markdown("""
    이번 탐구에서는 연속적인 수치인 건강수준 인지율을
    예측하는 회귀 분석을 중심으로 평가합니다.

    평균 모델과 선형회귀 모델을 비교하여
    의사 수라는 변수를 사용하는 것이 예측 오차를 줄이는지 확인합니다.
    """)

    st.divider()

    st.subheader("② 회귀 모델과 베이스라인 비교")

    X = df[[doctor]]
    y = df[health]

    loo = LeaveOneOut()

    # 평균 모델
    baseline_model = DummyRegressor(strategy="mean")

    # 선형회귀 모델
    regression_model = LinearRegression()

    # Leave-One-Out 교차검증 예측
    baseline_predictions = cross_val_predict(
        baseline_model,
        X,
        y,
        cv=loo
    )

    regression_predictions = cross_val_predict(
        regression_model,
        X,
        y,
        cv=loo
    )

    # 평가 지표 계산
    def calculate_metrics(actual, predicted):

        return {
            "R² (결정계수)": r2_score(actual, predicted),
            "MAE (평균 절대 오차)": mean_absolute_error(
                actual, predicted
            ),
            "RMSE (평균 제곱근 오차)": np.sqrt(
                mean_squared_error(actual, predicted)
            )
        }

    baseline_result = calculate_metrics(
        y,
        baseline_predictions
    )

    regression_result = calculate_metrics(
        y,
        regression_predictions
    )

    result_df = pd.DataFrame(
        [
            baseline_result,
            regression_result
        ],
        index=[
            "평균 모델(베이스라인)",
            "선형회귀 모델"
        ]
    )

    st.dataframe(
        result_df.style.format("{:.4f}"),
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 평가 지표 설명")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**R² (결정계수)**")
        st.write("""
        모델이 실제 데이터의 변동을 얼마나 설명하는지
        나타내는 지표입니다.
        일반적으로 높을수록 좋으며 음수가 나올 수도 있습니다.
        """)

    with col2:
        st.markdown("**MAE (평균 절대 오차)**")
        st.write("""
        예측값과 실제값의 차이를 절댓값으로 계산한
        평균입니다.
        값이 작을수록 예측 오차가 작습니다.
        """)

    with col3:
        st.markdown("**RMSE (평균 제곱근 오차)**")
        st.write("""
        예측 오차를 제곱한 뒤 평균을 내고
        제곱근을 취한 지표입니다.
        큰 오차에 더 민감하며 낮을수록 좋습니다.
        """)

    st.divider()

    st.subheader("④ 모델 성능 그래프")

    metric_names = [
        "R² (결정계수)",
        "MAE (평균 절대 오차)",
        "RMSE (평균 제곱근 오차)"
    ]

    fig5 = go.Figure()

    fig5.add_trace(
        go.Bar(
            x=metric_names,
            y=[
                baseline_result[name]
                for name in metric_names
            ],
            name="평균 모델"
        )
    )

    fig5.add_trace(
        go.Bar(
            x=metric_names,
            y=[
                regression_result[name]
                for name in metric_names
            ],
            name="선형회귀 모델"
        )
    )

    fig5.update_layout(
        title="평균 모델과 선형회귀 모델의 성능 비교",
        barmode="group",
        yaxis_title="평가 지표 값"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.caption(
        "R²는 높을수록 좋고 MAE와 RMSE는 낮을수록 좋습니다."
    )

    st.divider()

    st.subheader("⑤ 실제값과 예측값 비교")

    comparison_df = pd.DataFrame({
        "지역": df["지역"],
        "실제 건강수준": y.values,
        "평균 모델 예측": baseline_predictions,
        "선형회귀 예측": regression_predictions
    })

    st.dataframe(
        comparison_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    fig6 = go.Figure()

    fig6.add_trace(
        go.Bar(
            x=df["지역"],
            y=y,
            name="실제값"
        )
    )

    fig6.add_trace(
        go.Scatter(
            x=df["지역"],
            y=baseline_predictions,
            mode="lines+markers",
            name="평균 모델 예측"
        )
    )

    fig6.add_trace(
        go.Scatter(
            x=df["지역"],
            y=regression_predictions,
            mode="lines+markers",
            name="선형회귀 예측"
        )
    )

    fig6.update_layout(
        title="지역별 실제값과 모델 예측값",
        xaxis_title="지역",
        yaxis_title="건강수준 인지율(%)"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    st.divider()

    st.subheader("⑥ 분류 및 군집 평가 기준")

    st.markdown("""
    사진에 제시된 분류와 군집 평가 기준은 다음과 같습니다.

    **분류**

    | 평가 지표 | 의미 |
    |---|---|
    | 정확도 | 전체 예측 중 정답의 비율 |
    | 정밀도 | 특정 범주라고 예측한 것 중 실제 해당 범주의 비율 |
    | 재현율 | 실제 해당 범주 중 모델이 찾아낸 비율 |
    | F1 | 정밀도와 재현율의 조화평균 |
    | 베이스라인 | 가장 많은 범주만 예측하는 다수 범주 모델 |

    **군집**

    | 평가 지표 | 의미 |
    |---|---|
    | 실루엣 점수 | 군집 내부의 응집도와 군집 간 분리 정도 |
    | 엘보 방법 | 군집 수에 따른 관성의 변화를 확인하는 방법 |
    | k 선택 | 실루엣 점수와 엘보 그래프를 함께 고려 |

    현재 탐구의 종속변수는 연속적인 건강수준 인지율이므로
    회귀 분석이 주된 분석 방법입니다.

    분류는 건강수준을 범주로 나누는 별도 설계가 필요하고,
    군집은 지역을 비슷한 특성끼리 묶는 별도 탐색 분석입니다.

    현재 3개 지역만으로는 이 두 방법의 결과를 신뢰성 있게
    평가하기 어렵기 때문에 실제 모델을 실행하지 않습니다.
    """)

    


# ==========================================
# 5. 하단 안내
# ==========================================

st.divider()

st.caption(
    "주관적 건강수준 자료: 질병관리청 지역사회건강조사, 2021년"
)
