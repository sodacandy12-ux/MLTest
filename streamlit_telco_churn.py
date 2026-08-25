import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 페이지 기본 설정
st.set_page_config(page_title="통신 고객 이탈 예측 시스템", layout="wide")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔고딕 설정 및 마이너스 깨짐 방지
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# 1. 전처리 객체 및 학습 모델 로드
@st.cache_resource
def load_ml_objects():
    encoder = joblib.load('model/telco_encoder.joblib')
    scaler = joblib.load('model/telco_scaler.joblib')
    model = joblib.load('model/telco_model.joblib')
    meta = joblib.load('model/telco_features_meta.joblib')
    return encoder, scaler, model, meta


try:
    encoder, scaler, model, meta = load_ml_objects()
    numeric_cols = meta['numeric_cols']
    categorical_cols = meta['categorical_cols']
    binary_mappings = meta.get('binary_mappings', {})
    feature_names = meta['feature_names']
    decision_threshold = float(meta.get('decision_threshold', 0.50))
except FileNotFoundError:
    st.error("학습된 모델 파일(joblib)을 찾을 수 없습니다. 먼저 _make_telco_nb.ipynb에서 모델을 학습하고 저장해주세요.")
    st.stop()


# 시각화용 원본 데이터 로드
@st.cache_data
def load_raw_data():
    df = pd.read_csv('dataset/Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    return df


df = load_raw_data()


# ----------------- Streamlit UI 구성 -----------------
st.title("통신 고객 이탈 예측 시스템")
st.write("고객의 계약·요금·부가서비스 정보를 입력하면 학습된 분류 모델이 이탈 가능성을 예측합니다.")

main_tab1, main_tab2 = st.tabs(["이탈 위험도 시뮬레이터", "데이터 인사이트 대시보드"])

# ==================== TAB 1: 예측 시뮬레이터 ====================
with main_tab1:
    st.markdown("### 개별 고객 이탈 예측")
    st.write("아래에 고객 정보를 입력한 후 '예측 실행' 버튼을 눌러주세요.")

    col1, col2 = st.columns([2, 1.1])

    with col1:
        st.subheader("고객 정보 입력")

        with st.container(border=True):
            st.caption("기본 정보")
            sub1, sub2, sub3, sub4 = st.columns(4)
            with sub1:
                gender = st.selectbox("성별 (gender)", options=["Female", "Male"])
            with sub2:
                SeniorCitizen = st.selectbox(
                    "고령 여부 (SeniorCitizen)",
                    options=[0, 1],
                    format_func=lambda x: "해당 없음" if x == 0 else "고령",
                )
            with sub3:
                Partner = st.selectbox("배우자 여부 (Partner)", options=["No", "Yes"])
            with sub4:
                Dependents = st.selectbox("부양가족 여부 (Dependents)", options=["No", "Yes"])

            st.caption("이용 기간 / 요금")
            sub5, sub6, sub7 = st.columns(3)
            with sub5:
                tenure = st.slider("이용 개월 수 (tenure)", 0, 72, 12)
            with sub6:
                MonthlyCharges = st.number_input("월 요금 (MonthlyCharges)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            with sub7:
                TotalCharges = st.number_input("총 요금 (TotalCharges)", min_value=0.0, max_value=10000.0, value=float(tenure * MonthlyCharges), step=10.0)

            st.caption("전화 / 인터넷 서비스")
            sub8, sub9, sub10 = st.columns(3)
            with sub8:
                PhoneService = st.selectbox("전화 서비스 (PhoneService)", options=["No", "Yes"])
            with sub9:
                MultipleLines = st.selectbox("다중회선 (MultipleLines)", options=["No", "Yes", "No phone service"])
            with sub10:
                InternetService = st.selectbox("인터넷 (InternetService)", options=["DSL", "Fiber optic", "No"])

            st.caption("부가 서비스")
            sub11, sub12, sub13 = st.columns(3)
            with sub11:
                OnlineSecurity = st.selectbox("온라인 보안 (OnlineSecurity)", options=["No", "Yes", "No internet service"])
            with sub12:
                OnlineBackup = st.selectbox("온라인 백업 (OnlineBackup)", options=["No", "Yes", "No internet service"])
            with sub13:
                DeviceProtection = st.selectbox("기기 보호 (DeviceProtection)", options=["No", "Yes", "No internet service"])

            sub14, sub15, sub16 = st.columns(3)
            with sub14:
                TechSupport = st.selectbox("기술 지원 (TechSupport)", options=["No", "Yes", "No internet service"])
            with sub15:
                StreamingTV = st.selectbox("스트리밍 TV (StreamingTV)", options=["No", "Yes", "No internet service"])
            with sub16:
                StreamingMovies = st.selectbox("스트리밍 영화 (StreamingMovies)", options=["No", "Yes", "No internet service"])

            st.caption("계약 / 결제")
            sub17, sub18, sub19 = st.columns(3)
            with sub17:
                Contract = st.selectbox("계약 유형 (Contract)", options=["Month-to-month", "One year", "Two year"])
            with sub18:
                PaperlessBilling = st.selectbox("전자청구서 (PaperlessBilling)", options=["No", "Yes"])
            with sub19:
                PaymentMethod = st.selectbox(
                    "결제 방식 (PaymentMethod)",
                    options=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )

    with col2:
        st.subheader("예측 결과")
        st.write("")

        input_dict = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
        }
        input_df = pd.DataFrame([input_dict])

        # 학습 때 저장한 매핑으로 이진 범주를 0/1로 변환
        for col, mapping in binary_mappings.items():
            input_df[col] = input_df[col].map(mapping)
            if input_df[col].isna().any():
                st.error(f"이진 변수 '{col}'에 학습 시 없던 값이 입력되었습니다.")
                st.stop()
            input_df[col] = input_df[col].astype('int64')

        # 로드한 Encoder로 3개 이상 범주형 변수만 원핫인코딩
        input_cat = encoder.transform(input_df[categorical_cols])
        encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)

        # 수치형과 범주형(원핫인코딩) 컬럼 결합
        input_encoded = pd.concat([
            input_df[numeric_cols].reset_index(drop=True),
            pd.DataFrame(input_cat, columns=encoded_cat_cols),
        ], axis=1)

        # 학습 당시 컬럼 순서에 맞추기
        input_encoded = input_encoded[feature_names]

        # 스케일링 적용
        input_scaled = scaler.transform(input_encoded)

        with st.container(border=True):
            if st.button("이탈 여부 예측 실행", use_container_width=True, type="primary"):
                prediction_proba = model.predict_proba(input_scaled)[0]
                prediction = int(prediction_proba[1] >= decision_threshold)

                st.metric(label="예측 상태 결과", value="이탈 위험" if prediction == 1 else "유지 가능")
                st.caption(
                    f"운영 임계값 {decision_threshold:.2f} 적용 "
                    "(학습 OOF에서 Recall 70% 이상 조건으로 선택)"
                )
                st.markdown("---")

                if prediction == 1:
                    st.error(f"**이 고객은 이탈 가능성이 높습니다.**\n\n(이탈 확률: **{prediction_proba[1] * 100:.2f}%**)")
                else:
                    st.success(f"**이 고객은 서비스를 유지할 가능성이 높습니다.**\n\n(유지 확률: **{prediction_proba[0] * 100:.2f}%**)")

                st.caption("참고 지표이며, 실제 해지를 확정하는 값이 아닙니다.")
            else:
                st.info("왼쪽 정보를 채운 뒤 버튼을 누르면 예측 결과가 표시됩니다.")

# ==================== TAB 2: 데이터 인사이트 대시보드 ====================
with main_tab2:
    st.markdown("### 고객 데이터 시각화 및 인사이트")
    st.write("전체 데이터셋 분석을 기반으로 이탈과 관련된 주요 패턴을 확인합니다.")

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["AI 특성 중요도 분석", "계약·이용기간 분석", "요금 분포"])

    with sub_tab1:
        st.subheader("이탈에 영향을 주는 주요 피처")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if hasattr(model, "coef_"):
            st.caption("로지스틱 회귀 계수의 절댓값 상위 10개입니다. 양수는 이탈 확률 증가, 음수는 감소 방향이며 인과로 해석하지 않습니다.")
            scores = pd.Series(model.coef_[0], index=feature_names)
            top = scores.reindex(scores.abs().sort_values(ascending=False).head(10).index)
            sns.barplot(x=top, y=top.index, ax=ax, palette="vlag")
            ax.axvline(0, color="gray", linewidth=0.8)
            plt.title("Top 10 Standardized Coefficients", fontsize=12)
            plt.xlabel("Coefficient")
        else:
            st.caption("모델이 이탈 여부를 판단할 때 중요하게 본 상위 10개 변수입니다. 인과로 해석하지 않습니다.")
            scores = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
            sns.barplot(x=scores, y=scores.index, ax=ax, palette="viridis")
            plt.title("Top 10 Feature Importances", fontsize=12)
            plt.xlabel("Importance")
        st.pyplot(fig)
        plt.close(fig)

    with sub_tab2:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("계약 유형별 이탈 여부")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.countplot(x='Contract', hue='Churn', data=df, ax=ax, palette="Set2")
            plt.legend(title="이탈 여부", labels=["유지", "이탈"])
            plt.title("Contract vs Churn")
            st.pyplot(fig)
            plt.close(fig)

        with col_t2:
            st.subheader("이용 기간과 이탈")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.boxplot(x='Churn', y='tenure', data=df, ax=ax, palette="coolwarm")
            ax.set_xticklabels(["유지 (0)", "이탈 (1)"])
            plt.title("Tenure vs Churn")
            st.pyplot(fig)
            plt.close(fig)

    with sub_tab3:
        st.subheader("월 요금 분포")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', ax=ax, bins=30, palette="magma")
        plt.legend(title="이탈 여부", labels=["이탈", "유지"])
        plt.title("Monthly Charges Distribution")
        st.pyplot(fig)
        plt.close(fig)
