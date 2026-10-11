# 통신 고객 이탈 예측 시스템 (Telco Customer Churn)

고객의 계약·요금·부가서비스 정보를 입력하면 학습된 머신러닝 모델이 이탈 가능성을 예측하고, 데이터 인사이트를 함께 보여주는 Streamlit 웹 앱

**Python · scikit-learn · XGBoost · Streamlit · pandas · seaborn**

---

## 프로젝트 소개

통신사 고객 데이터(Telco Customer Churn, 7,043건 / 21개 컬럼)를 활용해 이탈(Churn) 여부를 예측하는 분류 모델을 학습하고, 이를 Streamlit 앱으로 서비스화한 프로젝트입니다.

- 노트북에서 전처리 → 모델 비교 → 하이퍼파라미터 튜닝 → 운영 임계값 선정 → 모델 저장까지 수행
- 저장된 모델·전처리 객체를 앱에서 불러와 개별 고객의 이탈 위험도를 실시간으로 예측
- 전체 데이터 기반 시각화 대시보드로 이탈 패턴 확인

---

## 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| ML | scikit-learn, XGBoost |
| Web App | Streamlit |
| Data / Visualization | pandas, NumPy, matplotlib, seaborn |
| Model Persistence | joblib |
| Environment | Jupyter Notebook |

---

## 주요 기능

### 1. 이탈 위험도 시뮬레이터
- 성별, 고령 여부, 이용 개월 수, 월/총 요금, 인터넷·부가서비스, 계약 유형, 결제 방식 등 19개 항목 입력
- 저장된 전처리 객체(이진 매핑 · OneHotEncoder · StandardScaler)를 동일하게 적용
- **이탈 확률**과 함께 "이탈 위험 / 유지 가능" 판정 결과 제공
- 학습 단계에서 선정한 운영 임계값(0.33) 적용

### 2. 데이터 인사이트 대시보드
- **AI 특성 중요도 분석** — 모델이 중요하게 본 상위 10개 변수 (인과 해석 아님)
- **계약·이용기간 분석** — 계약 유형별 이탈 여부, 이용 기간별 이탈 분포
- **요금 분포** — 월 요금 구간별 이탈/유지 분포

---

## 모델링 과정

| 단계 | 내용 |
|---|---|
| 전처리 | `TotalCharges` 숫자 변환(공백 → 0), `customerID` 제거, `Churn` 0/1 변환 |
| 인코딩 | 이진 범주(5개)는 0/1 매핑, 3개 이상 범주(10개)는 `OneHotEncoder(drop='first')` |
| 스케일링 | `StandardScaler` |
| 데이터 분할 | Train/Test = 8:2 (`stratify=y`, `random_state=42`) |
| 모델 비교 | Logistic Regression / Random Forest / XGBoost |
| 튜닝 | `GridSearchCV` (cv=3, scoring=`roc_auc`) |
| 최종 모델 | 테스트 ROC-AUC가 가장 높은 모델 → **XGBoost** |
| 임계값 선정 | 학습 데이터 3-Fold OOF에서 **Recall ≥ 0.70** 조건 하에 F1 최대 → **0.33** |

> 이탈 고객을 놓치는 비용이 크다고 보고 Recall을 우선하도록 기본값(0.5) 대신 낮은 임계값을 사용했습니다.
> 임계값은 테스트셋이 아닌 학습 데이터의 OOF 예측으로 선정하여 데이터 누수를 방지했습니다.

---

## 프로젝트 구조

```text
MLTest/
├── streamlit_telco_churn.py     # Streamlit 앱
├── _make_telco_nb.ipynb         # 전처리·모델 학습·평가·저장 노트북
├── requirements.txt             # Python 의존성
├── packages.txt                 # 시스템 패키지 (한글 폰트: fonts-nanum)
│
├── dataset/
│   └── Telco-Customer-Churn.csv
│
└── model/
    ├── telco_model.joblib           # 학습된 XGBoost 모델
    ├── telco_encoder.joblib         # OneHotEncoder
    ├── telco_scaler.joblib          # StandardScaler
    └── telco_features_meta.joblib   # 컬럼 목록·이진 매핑·임계값 등 메타 정보
```

---

## 로컬 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/sodacandy12-ux/MLTest.git
cd MLTest
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 앱 실행

```bash
streamlit run streamlit_telco_churn.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

### (선택) 모델 재학습

`_make_telco_nb.ipynb`를 순서대로 실행하면 `model/` 폴더의 joblib 파일이 다시 생성됩니다.

---

## 참고 사항

- 시각화의 한글 표시를 위해 나눔고딕(`NanumGothic`)을 사용합니다. 배포 환경(Streamlit Cloud 등)에서는 `packages.txt`의 `fonts-nanum`으로 설치되며, 로컬에서 글자가 깨지면 나눔고딕을 설치해 주세요.
- 예측 결과는 참고 지표이며 실제 해지를 확정하는 값이 아닙니다.
- 특성 중요도는 변수와 이탈 간의 연관성을 보여줄 뿐 인과관계를 의미하지 않습니다.

---

## 향후 개선 방향

- SHAP 기반 개별 고객 예측 근거 설명
- 이탈 위험 고객 대상 맞춤 유지 전략 제안
- CSV 업로드를 통한 다수 고객 일괄 예측
