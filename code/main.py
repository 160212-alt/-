
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="지역별 의료환경 데이터 분석",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 지역별 의료환경과 건강수준 분석")

st.markdown("""
**KOSIS 통계 활용 데이터 과학 탐구**

지역별 인구 1만 명당 의사 수와 주관적 건강수준의 관계를 분석합니다.
""")

st.divider()


# ==========================================
# 2. 데이터 직접 입력
# ==========================================

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
        52.9, 46.9, 47.4, 45.6, 48.6,
        55.2, 48.6, 48.9, 48.6, 48.0,
        47.8, 48.4, 43.9, 48.7, 47.4,
        45.6, 52.5
    ],

    "총인구": [
        9509000, 3350000, 2385000, 2948000,
        1442000, 1452000, 1122000, 372000,
        13565000, 1538000, 1597000, 2119000,
        1787000, 1833000, 2627000, 3314000,
        677000
    ]
}

df = pd.DataFrame(data)


# ==========================================
# 3. 공통 변수 설정
# ==========================================

doctor = "인구 1만 명당 의사 수"
health = "주관적 건강수준 인지율(%)"
population = "총인구"

numeric_cols = [doctor, health, population]


# ==========================================
# 4. 상단 메뉴 구성
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
    3. 지역별 총인구와 건강수준 사이에는 어떤 관계가 있을까?
    4. 의사 수를 이용하여 주관적 건강수준을 예측할 수 있을까?
    """)

    st.divider()

    st.subheader("탐구 가설")

    st.write("""
    인구 1만 명당 의사 수가 많은 지역일수록
    의료서비스를 이용할 수 있는 기회가 많아져
    주관적 건강수준 인지율도 높게 나타날 것이다.
    """)

    st.subheader("분석 목표")

    st.write("""
    전국 17개 시도의 의료환경과 건강수준 데이터를 활용하여
    두 변수 사이의 상관관계를 분석한다.

    또한 선형회귀 모델을 활용하여 의사 수에 따른
    건강수준 예측 가능성을 확인한다.
    """)

    st.divider()

    st.subheader("데이터 기본 정보")

    c1, c2, c3 = st.columns(3)

    c1.metric("분석 지역", "17개")
    c2.metric("분석 연도", "2021년")
    c3.metric("분석 변수", "3개")


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

    st.caption(
        "아래 수치는 이전에 작성한 데이터 초안에서 가져온 값이며, "
        "KOSIS 원자료와 대조 검증이 완료된 자료는 아닙니다."
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

    st.subheader("③ 지역별 건강수준 비교")

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
        yaxis_title="건강수준 인지율(%)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("④ 변수 간 관계 분석")

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
            "피어슨 상관계수",
            f"{corr:.3f}"
        )

        fig3 = px.scatter(
            df,
            x=x_var,
            y=y_var,
            text="지역",
            hover_name="지역",
            title=f"{x_var}와 {y_var}의 관계"
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

    else:
        st.warning("서로 다른 변수를 선택해 주세요.")

    st.divider()

    st.subheader("⑤ 전체 변수 상관관계 히트맵")

    corr_matrix = df[numeric_cols].corr()

    fig4 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="변수 간 상관관계"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.caption(
        "상관계수는 -1부터 1까지의 값을 가지며, "
        "절댓값이 클수록 선형 상관관계가 강합니다."
    )


# ==========================================
# TAB 3. 기준선 모델
# ==========================================

with tab3:

    st.header("3. 기준선 모델")

    st.subheader("기준선 모델이란?")

    st.write("""
    기준선 모델은 복잡한 예측 방법을 사용하지 않고
    모든 지역의 건강수준을 전체 지역의 평균값으로 예측하는 모델이다.

    이후 선형회귀 모델의 성능을 비교하기 위한 기준으로 활용한다.
    """)

    baseline = df[health].mean()

    st.metric(
        "전체 지역 건강수준 평균",
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
            name="실제 건강수준"
        )
    )

    fig5.add_trace(
        go.Scatter(
            x=baseline_df["지역"],
            y=baseline_df["기준선 예측값"],
            name="기준선 예측값",
            mode="lines",
            line=dict(dash="dash")
        )
    )

    fig5.update_layout(
        title="지역별 실제 건강수준과 기준선 비교",
        xaxis_title="지역",
        yaxis_title="건강수준 인지율(%)"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.divider()

    st.subheader("기준선 모델의 의미")

    st.write(f"""
    모든 지역의 건강수준을 {baseline:.2f}%로 예측한다.

    실제 건강수준과 예측값의 차이가 작을수록
    기준선 모델의 예측 오차가 작아진다.

    이후 모델 평가 메뉴에서 선형회귀 모델과 비교한다.
    """)


# ==========================================
# TAB 4. 모델 평가
# ==========================================

with tab4:

    st.header("4. 모델 평가")

    st.subheader("선형회귀 모델")

    st.write("""
    인구 1만 명당 의사 수를 독립변수(X)로 설정하고,
    주관적 건강수준 인지율을 종속변수(Y)로 설정한다.

    의사 수를 이용하여 건강수준을 예측하는
    단순 선형회귀 모델을 평가한다.
    """)

    X = df[[doctor]]
    y = df[health]

    model = LinearRegression()

    # Leave-One-Out 교차검증
    loo = LeaveOneOut()

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

    # 기준선도 같은 교차검증 방식으로 평가
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

    st.subheader("① 모델 성능 평가 지표")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("R²", f"{r2:.3f}")

    st.caption("""
    MAE와 RMSE는 예측값과 실제값의 차이를 나타낸다.
    두 값은 낮을수록 예측 오차가 작다.
    R²는 예측 성능을 평가하는 지표이며, 음수가 나올 수도 있다.
    """)

    st.divider()

    st.subheader("② 기준선과 회귀모델 비교")

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
        title="기준선 모델과 선형회귀 모델의 오차 비교"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 실제값과 예측값 비교")

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

    st.subheader("④ 최종 모델 평가")

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
    **해석 시 주의사항**

    분석 대상이 전국 17개 시도로 제한되어 있으므로
    표본 수가 적다는 한계가 있다.

    또한 의료자원과 건강수준 사이의 상관관계가
    반드시 인과관계를 의미하는 것은 아니다.
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
    "데이터 출처: KOSIS 국가통계포털을 활용한 탐구용 데이터 초안. "
    원자료와 수치 대조가 필요합니다."
)
