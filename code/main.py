
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------
# 1. 기본 설정
# --------------------------------

st.set_page_config(
    page_title="지역별 의료 환경 데이터 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 지역별 의료 환경과 건강수준 분석")
st.caption(
    "KOSIS 데이터를 활용한 지역별 만 명당 의사 수와 "
    "주관적 건강수준 인지율의 관계 분석"
)

st.markdown("""
이 프로젝트는 지역별 의료 인력 수준과 주민의 주관적 건강수준 사이에
어떤 관계가 있는지 통계 및 머신러닝 방법으로 분석합니다.
""")

# --------------------------------
# 2. 데이터 기본 구조
# --------------------------------

regions = [
    "서울", "부산", "대구", "인천", "광주",
    "대전", "울산", "세종", "경기", "강원",
    "충북", "충남", "전북", "전남", "경북",
    "경남", "제주"
]

template = pd.DataFrame({
    "지역": regions,
    "만명당 의사 수": [np.nan] * len(regions),
    "주관적 건강수준 인지율": [np.nan] * len(regions)
})

# --------------------------------
# 3. 데이터 업로드
# --------------------------------

st.sidebar.header("데이터 설정")

uploaded_file = st.sidebar.file_uploader(
    "KOSIS CSV 파일 업로드",
    type=["csv"]
)

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}

    for col in df.columns:
        name = col.replace(" ", "")

        if name in ["지역", "시도", "행정구역"]:
            rename_map[col] = "지역"

        elif name in [
            "만명당의사수",
            "인구10,000명당의사수",
            "의사수"
        ]:
            rename_map[col] = "만명당 의사 수"

        elif name in [
            "주관적건강수준인지율",
            "주관적건강수준",
            "건강수준인지율"
        ]:
            rename_map[col] = "주관적 건강수준 인지율"

    return df.rename(columns=rename_map)


if uploaded_file is not None:
    try:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="cp949")

        df = normalize_columns(raw_df)

        required = [
            "지역",
            "만명당 의사 수",
            "주관적 건강수준 인지율"
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(
                "CSV에서 필요한 열을 찾지 못했습니다: "
                + ", ".join(missing)
            )
            st.info(
                "열 이름을 '지역', '만명당 의사 수', "
                "'주관적 건강수준 인지율'로 맞춰 주세요."
            )
            df = template.copy()
        else:
            st.sidebar.success("CSV 파일을 불러왔습니다.")

    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        df = template.copy()

else:
    df = template.copy()

# --------------------------------
# 4. 데이터 편집
# --------------------------------

st.subheader("데이터 확인 및 수정")

st.write(
    "KOSIS에서 받은 수치를 입력하거나 업로드한 데이터를 수정할 수 있습니다."
)

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor"
)

# 숫자형 변환
for col in edited_df.columns:
    if col != "지역":
        edited_df[col] = pd.to_numeric(
            edited_df[col],
            errors="coerce"
        )

# 유효한 데이터만 추출
required_columns = [
    "지역",
    "만명당 의사 수",
    "주관적 건강수준 인지율"
]

if all(c in edited_df.columns for c in required_columns):
    data = edited_df.dropna(
        subset=[
            "만명당 의사 수",
            "주관적 건강수준 인지율"
        ]
    ).copy()
else:
    data = pd.DataFrame(columns=required_columns)

data = data.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        "만명당 의사 수",
        "주관적 건강수준 인지율"
    ]
)

# --------------------------------
# 5. CSV 다운로드
# --------------------------------

st.download_button(
    "현재 데이터 CSV 다운로드",
    data=edited_df.to_csv(
        index=False,
        encoding="utf-8-sig"
    ),
    file_name="kosis_health_data.csv",
    mime="text/csv"
)

# --------------------------------
# 6. 탭 구성
# --------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "프로젝트 소개",
    "데이터 입력",
    "데이터 분석",
    "기준선",
    "모델 평가"
])

# --------------------------------
# TAB 1. 프로젝트 소개
# --------------------------------

with tab1:
    st.header("탐구 주제")

    st.subheader(
        "지역별 만 명당 의사 수와 주관적 건강수준 인지율의 관계"
    )

    st.write("""
    지역별 의료 인력의 차이가 주민들이 스스로 인식하는
    건강수준과 어떤 관계를 가지는지 알아보는 탐구입니다.

    **핵심 탐구 질문**

    만 명당 의사 수가 많은 지역일수록 주관적 건강수준
    인지율도 높게 나타날까?
    """)

    st.subheader("분석 목표")

    st.markdown("""
    1. 지역별 의료 인력과 건강수준의 차이를 파악한다.
    2. 두 변수 사이의 상관관계를 분석한다.
    3. 선형회귀 모델을 이용해 건강수준 인지율을 예측한다.
    4. 평균 모델과 선형회귀 모델의 성능을 비교한다.
    5. 분류와 군집 분석을 추가로 적용해 데이터 분석 방법의 차이를 알아본다.
    """)

    st.info(
        "상관관계가 발견되더라도 의사 수가 건강수준을 "
        "직접적으로 높였다고 단정할 수는 없습니다."
    )

# --------------------------------
# TAB 2. 데이터 입력
# --------------------------------

with tab2:
    st.header("데이터 구성")

    st.write("현재 분석에 사용되는 데이터의 구조입니다.")

    st.dataframe(data, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("유효 데이터 수", f"{len(data)}개")

    if len(data) > 0:
        col2.metric(
            "평균 의사 수",
            f"{data['만명당 의사 수'].mean():.2f}명"
        )

        col3.metric(
            "평균 건강수준 인지율",
            f"{data['주관적 건강수준 인지율'].mean():.2f}%"
        )

    st.subheader("데이터 설명")

    st.markdown("""
    | 변수 | 의미 | 자료 형태 |
    |---|---|---|
    | 지역 | 통계가 조사된 지역 | 범주형 |
    | 만명당 의사 수 | 인구 1만 명당 의사 수 | 수치형 |
    | 주관적 건강수준 인지율 | 자신의 건강이 좋다고 인식하는 비율 | 수치형 |
    """)

    st.warning(
        "현재 코드는 두 핵심 변수만으로 회귀·분류·군집 분석을 진행합니다. "
        "추가 변수를 확보했다면 군집 분석에 함께 활용할 수 있습니다."
    )

# --------------------------------
# TAB 3. 데이터 분석
# --------------------------------

with tab3:
    st.header("데이터 분석")

    if len(data) < 2:
        st.warning("분석하려면 유효한 데이터가 최소 2개 필요합니다.")

    else:
        x = data["만명당 의사 수"]
        y = data["주관적 건강수준 인지율"]

        correlation = x.corr(y)

        col1, col2 = st.columns(2)

        col1.metric(
            "상관계수",
            f"{correlation:.3f}"
        )

        if correlation > 0:
            relation = "양의 상관관계"
        elif correlation < 0:
            relation = "음의 상관관계"
        else:
            relation = "상관관계가 거의 없음"

        col2.metric("관계 방향", relation)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(x, y, s=70)

        if len(data) >= 2 and x.nunique() >= 2:
            model = LinearRegression()
            model.fit(x.to_frame(), y)

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = model.predict(
                pd.DataFrame({"만명당 의사 수": x_line})
            )

            ax.plot(x_line, y_line)

        for _, row in data.iterrows():
            ax.annotate(
                str(row["지역"]),
                (
                    row["만명당 의사 수"],
                    row["주관적 건강수준 인지율"]
                )
            )

        ax.set_xlabel("만 명당 의사 수")
        ax.set_ylabel("주관적 건강수준 인지율 (%)")
        ax.set_title("의사 수와 주관적 건강수준의 관계")

        st.pyplot(fig)

        st.subheader("상관관계 해석")

        st.write(f"두 변수의 상관계수는 **{correlation:.3f}**입니다.")

        st.write("""
        상관계수는 두 변수 사이의 선형적인 관계의 방향과 정도를
        나타냅니다. 다만 상관관계만으로 인과관계를 증명할 수는 없습니다.
        """)

# --------------------------------
# TAB 4. 기준선
# --------------------------------

with tab4:
    st.header("베이스라인 모델")

    st.write("""
    베이스라인은 복잡한 모델을 사용하지 않았을 때의
    기본적인 예측 성능입니다.
    """)

    st.subheader("1. 회귀의 베이스라인: 평균 모델")

    st.write("""
    모든 지역의 주관적 건강수준 인지율을 학습 데이터의 평균값으로
    예측하는 단순한 모델입니다.

    선형회귀 모델이 이 평균 모델보다 좋은 성능을 보이는지 비교합니다.
    """)

    if len(data) > 0:
        st.metric(
            "전체 데이터 평균 건강수준 인지율",
            f"{data['주관적 건강수준 인지율'].mean():.2f}%"
        )

    st.subheader("2. 분류의 베이스라인: 다수 범주 찍기")

    st.write("""
    건강수준 인지율을 중앙값 기준으로 높음과 낮음으로 나눈 뒤,
    학습 데이터에서 더 많은 범주를 무조건 예측하는 방식입니다.

    실제 분류 모델이 이보다 정확한지 비교합니다.
    """)

    st.subheader("3. 군집 분석")

    st.write("""
    군집 분석에는 정답이 없기 때문에 다수 범주 찍기와 같은
    분류 베이스라인을 사용하지 않습니다.

    실루엣 점수와 엘보 방법을 이용해 적절한 군집 수를 살펴봅니다.
    """)

# --------------------------------
# TAB 5. 모델 평가
# --------------------------------

with tab5:
    st.header("모델 평가")

    st.write("""
    회귀, 분류, 군집 분석을 각각 적용하고
    각 방법에 알맞은 평가 지표를 확인합니다.
    """)

    if len(data) < 4:
        st.warning(
            "모델 평가에는 최소 4개 이상의 유효 데이터가 필요합니다. "
            "가능하면 17개 시도 전체 데이터를 입력해 주세요."
        )

    else:
        # -------------------------
        # A. 회귀 모델 평가
        # -------------------------

        st.subheader("A. 회귀 모델 평가")

        st.write("""
        선형회귀 모델과 평균 예측 모델을 비교합니다.
        평가에는 Leave-One-Out 교차검증을 사용합니다.
        """)

        X = data[["만명당 의사 수"]]
        y = data["주관적 건강수준 인지율"]

        loo = LeaveOneOut()

        regression_model = LinearRegression()
        baseline_model = DummyRegressor(strategy="mean")

        pred_reg = cross_val_predict(
            regression_model,
            X,
            y,
            cv=loo
        )

        pred_base = cross_val_predict(
            baseline_model,
            X,
            y,
            cv=loo
        )

        def regression_metrics(actual, predicted):
            return {
                "R²": r2_score(actual, predicted),
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, predicted)
                )
            }

        reg_result = regression_metrics(y, pred_reg)
        base_result = regression_metrics(y, pred_base)

        regression_table = pd.DataFrame(
            [base_result, reg_result],
            index=["평균 모델(베이스라인)", "선형회귀 모델"]
        )

        st.dataframe(
            regression_table.style.format("{:.4f}"),
            use_container_width=True
        )

        st.markdown("""
        **평가지표의 의미**

        - R²: 실제값의 변동을 모델이 얼마나 설명하는지 나타냅니다.
          높을수록 좋으며 음수가 나올 수도 있습니다.
        - MAE: 예측값과 실제값의 평균적인 차이입니다.
          낮을수록 좋습니다.
        - RMSE: 예측 오차를 제곱한 뒤 평균을 내고 제곱근을 취한 값입니다.
          큰 오차에 더 민감합니다.
        """)

        fig, ax = plt.subplots(figsize=(8, 4))

        metric_names = ["R²", "MAE", "RMSE"]
        x_pos = np.arange(len(metric_names))
        width = 0.35

        ax.bar(
            x_pos - width / 2,
            [base_result[m] for m in metric_names],
            width,
            label="평균 모델"
        )

        ax.bar(
            x_pos + width / 2,
            [reg_result[m] for m in metric_names],
            width,
            label="선형회귀 모델"
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names)
        ax.set_title("베이스라인과 선형회귀 성능 비교")
        ax.legend()

        st.pyplot(fig)

        st.write("**예측 결과 비교**")

        comparison = pd.DataFrame({
            "지역": data["지역"].values,
            "실제값": y.values,
            "평균 모델 예측": pred_base,
            "선형회귀 예측": pred_reg
        })

        st.dataframe(
            comparison.round(3),
            use_container_width=True
        )

        if reg_result["MAE"] < base_result["MAE"]:
            st.success(
                "선형회귀 모델의 MAE가 평균 모델보다 낮습니다. "
                "이번 평가에서는 선형회귀가 평균 모델보다 "
                "작은 예측 오차를 보였습니다."
            )
        else:
            st.warning(
                "선형회귀 모델이 평균 모델보다 좋은 MAE를 "
                "보이지 못했습니다. 단순한 평균 예측보다 "
                "성능이 낮을 수 있습니다."
            )

        # -------------------------
        # B. 분류 모델 평가
        # -------------------------

        st.divider()
        st.subheader("B. 분류 모델 평가")

        st.write("""
        주관적 건강수준 인지율을 중앙값 기준으로
        '높음'과 '낮음'으로 나누어 분류합니다.

        로지스틱 회귀 모델과 다수 범주 예측 모델을 비교합니다.
        """)

        median_value = y.median()

        labels = (y >= median_value).astype(int)

        class_counts = labels.value_counts()

        if len(class_counts) < 2 or class_counts.min() < 2:
            st.warning(
                "분류를 평가하려면 높음과 낮음 범주가 각각 "
                "최소 2개 이상 필요합니다."
            )

        else:
            classifier = LogisticRegression(
                max_iter=1000,
                class_weight=None
            )

            dummy_classifier = DummyClassifier(
                strategy="most_frequent"
            )

            pred_class = cross_val_predict(
                classifier,
                X,
                labels,
                cv=loo
            )

            pred_class_base = cross_val_predict(
                dummy_classifier,
                X,
                labels,
                cv=loo
            )

            def classification_metrics(actual, predicted):
                return {
                    "정확도": accuracy_score(actual, predicted),
                    "정밀도": precision_score(
                        actual, predicted, zero_division=0
                    ),
                    "재현율": recall_score(
                        actual, predicted, zero_division=0
                    ),
                    "F1": f1_score(
                        actual, predicted, zero_division=0
                    )
                }

            class_base_result = classification_metrics(
                labels, pred_class_base
            )

            class_result = classification_metrics(
                labels, pred_class
            )

            class_table = pd.DataFrame(
                [class_base_result, class_result],
                index=["다수 범주 찍기", "로지스틱 회귀"]
            )

            st.dataframe(
                class_table.style.format("{:.4f}"),
                use_container_width=True
            )

            st.markdown("""
            **평가지표의 의미**

            - 정확도: 전체 예측 중 맞힌 비율
            - 정밀도: 높음이라고 예측한 것 중 실제로 높은 비율
            - 재현율: 실제로 높은 지역 중 찾아낸 비율
            - F1: 정밀도와 재현율의 조화평균
            """)

            st.info(
                f"분류 기준: 주관적 건강수준 인지율 "
                f"{median_value:.2f}% 이상은 높음, 미만은 낮음"
            )

            st.caption(
                "이 분류는 분석을 위한 상대적인 구분이며, "
                "건강수준의 공식적인 등급을 의미하지 않습니다."
            )

        # -------------------------
        # C. 군집 분석 평가
        # -------------------------

        st.divider()
        st.subheader("C. 군집 분석 평가")

        st.write("""
        의사 수와 건강수준 인지율을 이용해 지역을 비슷한 특성끼리
        묶습니다.

        실루엣 점수와 엘보 방법으로 군집 수를 비교합니다.
        """)

        cluster_features = data[
            [
                "만명당 의사 수",
                "주관적 건강수준 인지율"
            ]
        ]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_features)

        max_k = min(6, len(data) - 1)

        silhouette_results = []
        inertia_results = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            cluster_labels = kmeans.fit_predict(scaled_data)

            score = silhouette_score(
                scaled_data,
                cluster_labels
            )

            silhouette_results.append((k, score))
            inertia_results.append((k, kmeans.inertia_))

        if silhouette_results:
            best_k = max(
                silhouette_results,
                key=lambda item: item[1]
            )[0]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**실루엣 점수 비교**")

                silhouette_df = pd.DataFrame(
                    silhouette_results,
                    columns=["군집 수 k", "실루엣 점수"]
                )

                st.dataframe(
                    silhouette_df.round(4),
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                ax.plot(
                    silhouette_df["군집 수 k"],
                    silhouette_df["실루엣 점수"],
                    marker="o"
                )

                ax.set_xlabel("군집 수 k")
                ax.set_ylabel("실루엣 점수")
                ax.set_title("군집 수에 따른 실루엣 점수")

                st.pyplot(fig)

            with col2:
                st.write("**엘보 방법**")

                inertia_df = pd.DataFrame(
                    inertia_results,
                    columns=["군집 수 k", "관성(Inertia)"]
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                ax.plot(
                    inertia_df["군집 수 k"],
                    inertia_df["관성(Inertia)"],
                    marker="o"
                )

                ax.set_xlabel("군집 수 k")
                ax.set_ylabel("관성(Inertia)")
                ax.set_title("엘보 방법")

                st.pyplot(fig)

            st.success(
                f"실루엣 점수가 가장 높은 군집 수는 k={best_k}입니다."
            )

            st.write("""
            **k를 선택한 이유**

            실루엣 점수는 각 데이터가 자기 군집 안에서는 얼마나
            가까우며 다른 군집과는 얼마나 떨어져 있는지 평가합니다.

            따라서 실루엣 점수가 가장 높은 k를 우선적인 후보로
            선택했습니다. 다만 엘보 그래프의 꺾이는 지점과
            함께 확인해야 합니다.
            """)

            final_model = KMeans(
                n_clusters=best_k,
                random_state=42,
                n_init=10
            )

            final_labels = final_model.fit_predict(scaled_data)

            cluster_result = data[
                [
                    "지역",
                    "만명당 의사 수",
                    "주관적 건강수준 인지율"
                ]
            ].copy()

            cluster_result["군집"] = final_labels + 1

            st.write("**지역별 군집 결과**")

            st.dataframe(
                cluster_result,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            scatter = ax.scatter(
                data["만명당 의사 수"],
                data["주관적 건강수준 인지율"],
                c=final_labels,
                cmap="viridis",
                s=80
            )

            for i, row in data.reset_index(drop=True).iterrows():
                ax.annotate(
                    str(row["지역"]),
                    (
                        row["만명당 의사 수"],
                        row["주관적 건강수준 인지율"]
                    )
                )

            ax.set_xlabel("만 명당 의사 수")
            ax.set_ylabel("주관적 건강수준 인지율 (%)")
            ax.set_title(f"K-means 군집 분석 (k={best_k})")

            st.pyplot(fig)

        else:
            st.warning(
                "군집 분석에 필요한 데이터가 부족합니다."
            )

        # -------------------------
        # D. 종합 평가
        # -------------------------

        st.divider()
        st.subheader("D. 종합 평가")

        st.markdown("""
        | 분석 방법 | 주요 평가지표 | 베이스라인 또는 기준 |
        |---|---|---|
        | 회귀 | R², MAE, RMSE | 평균 모델 |
        | 분류 | 정확도, 정밀도, 재현율, F1 | 다수 범주 찍기 |
        | 군집 | 실루엣 점수, 관성 | 엘보 방법과 k 비교 |
        """)

        st.write("""
        회귀는 연속적인 수치를 예측하기 때문에 R²와 MAE를 활용합니다.

        분류는 데이터를 범주로 구분하기 때문에 정확도, 정밀도,
        재현율, F1을 활용합니다.

        군집은 정답이 없는 데이터를 비슷한 특성끼리 묶는 방법이므로
        실루엣 점수와 엘보 방법을 이용해 군집의 적절성을 평가합니다.
        """)

        st.warning("""
        지역별 데이터는 표본 수가 많지 않으므로 평가 결과를
        일반화하는 데 한계가 있습니다. 특히 분류와 군집 결과는
        탐색적 분석으로 해석하는 것이 적절합니다.
        """)
