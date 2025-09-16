import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------
# 페이지 기본 설정
# -------------------------
st.set_page_config(
    page_title="키 분포 분석(고2)",
    page_icon="📊",
    layout="centered"
)

st.title("📊 고2 키 분포 분석 웹앱")
st.caption("파일: hist_heights.csv (같은 폴더)")

# -------------------------
# 데이터 로드
# -------------------------
CSV_NAME = "hist_heights.csv"

def load_data(path: str) -> pd.Series:
    df = pd.read_csv(path)
    # CSV는 Height_cm 컬럼 하나를 포함: 실수(float) cm 단위
    col = "Height_cm"
    if col not in df.columns:
        raise ValueError(f"'{col}' 컬럼을 찾을 수 없습니다. CSV 컬럼을 확인하세요.")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        raise ValueError("유효한 숫자 데이터가 없습니다.")
    return s

data = None
if os.path.exists(CSV_NAME):
    try:
        data = load_data(CSV_NAME)
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}")
else:
    st.warning("같은 폴더에 hist_heights.csv가 없습니다. 아래에서 파일을 업로드하세요.")
    up = st.file_uploader("hist_heights.csv 업로드", type=["csv"])
    if up is not None:
        try:
            data = load_data(up)
        except Exception as e:
            st.error(f"업로드 파일 처리 오류: {e}")

if data is None:
    st.stop()

# -------------------------
# 사이드바 옵션
# -------------------------
st.sidebar.header("⚙️ 그래프 옵션")
bin_count = st.sidebar.slider("막대(구간) 개수", min_value=5, max_value=25, value=12, step=1)
show_labels = st.sidebar.checkbox("막대 위에 빈도 레이블 표시", value=False)

# -------------------------
# 통계 요약
# -------------------------
count = int(data.count())
mean = float(data.mean())
median = float(data.median())
std = float(data.std(ddof=1))
min_v = float(data.min())
max_v = float(data.max())
q1 = float(data.quantile(0.25))
q3 = float(data.quantile(0.75))
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data < lower_bound) | (data > upper_bound)]
skewness = float(data.skew())

st.subheader("📌 기초 통계")
c1, c2, c3 = st.columns(3)
c1.metric("표본수", f"{count}명")
c2.metric("평균(Mean)", f"{mean:.1f} cm")
c3.metric("중앙값(Median)", f"{median:.1f} cm")

c4, c5, c6 = st.columns(3)
c4.metric("표준편차(Std)", f"{std:.2f} cm")
c5.metric("최솟값", f"{min_v:.1f} cm")
c6.metric("최댓값", f"{max_v:.1f} cm")

c7, c8, c9 = st.columns(3)
c7.metric("Q1(하위25%)", f"{q1:.1f} cm")
c8.metric("Q3(상위25%)", f"{q3:.1f} cm")
c9.metric("IQR", f"{iqr:.1f} cm")

st.markdown(f"- **이상치 기준**: {lower_bound:.1f} cm 미만 또는 {upper_bound:.1f} cm 초과")
st.markdown(f"- **이상치 개수**: {len(outliers)}명")
if 0 < len(outliers) <= 10:
    st.dataframe(pd.DataFrame({"Outlier(cm)": outliers.sort_values().values}))
else:
    if len(outliers) > 10:
        st.caption("이상치가 10명 초과이므로 목록은 생략했습니다.")

st.caption(f"왜도(Skewness): {skewness:.3f} (0에 가까우면 대체로 대칭)")

# -------------------------
# 히스토그램(막대 그래프) 데이터 만들기
# -------------------------
bins = np.linspace(data.min(), data.max(), bin_count + 1)
counts, edges = np.histogram(data, bins=bins)
bin_left = edges[:-1]
bin_right = edges[1:]

# 구간 라벨 문자열 예: "165.0–170.0"
labels = [f"{l:.1f}–{r:.1f}" for l, r in zip(bin_left, bin_right)]
hist_df = pd.DataFrame({
    "구간": labels,
    "빈도": counts,
    "left": bin_left,
    "right": bin_right
})

# -------------------------
# 막대 그래프 (Altair)
# -------------------------
st.subheader("📊 키 분포 막대 그래프(히스토그램)")
base = alt.Chart(hist_df).mark_bar().encode(
    x=alt.X("구간:O", title="키 구간 (cm)", sort=None),
    y=alt.Y("빈도:Q", title="인원수"),
    tooltip=[
        alt.Tooltip("구간:O", title="구간"),
        alt.Tooltip("빈도:Q", title="인원수"),
        alt.Tooltip("left:Q", title="구간시작(cm)", format=".1f"),
        alt.Tooltip("right:Q", title="구간끝(cm)", format=".1f"),
    ]
).properties(height=360)

chart = base

if show_labels:
    text = base.mark_text(dy=-8).encode(text="빈도:Q")
    chart = base + text

st.altair_chart(chart, use_container_width=True)

# -------------------------
# 다운로드: 구간별 빈도 CSV
# -------------------------
st.download_button(
    label="구간별 빈도 CSV 다운로드",
    data=hist_df[["구간", "빈도"]].to_csv(index=False).encode("utf-8-sig"),
    file_name="height_histogram_bins.csv",
    mime="text/csv"
)

# -------------------------
# 해석 가이드 (고2 눈높이)
# -------------------------
st.subheader("🧠 해석 가이드 (고2용)")
st.markdown("""
1) **가운데(중심)와 퍼짐(분산)**  
- 평균이 **{mean:.1f} cm**, 중앙값이 **{median:.1f} cm** 입니다. 평균과 중앙값이 비슷하면 대체로 **대칭적인 분포**로 볼 수 있어요.  
- 표준편차가 **{std:.2f} cm**이므로, 대부분 학생의 키는 평균에서 ±1표준편차(약 {low1:.1f}~{high1:.1f} cm) 범위에 몰려 있을 가능성이 큽니다.

2) **분위수와 IQR**  
- 하위 25%(Q1): **{q1:.1f} cm**, 상위 25%(Q3): **{q3:.1f} cm**  
- IQR(중간 50% 범위)은 **{iqr:.1f} cm**로, **중앙에 있는 50% 학생의 키가 이 범위에 분포**합니다.

3) **이상치(outlier)**  
- IQR을 기준으로 **{lower:.1f} cm 미만** 또는 **{upper:.1f} cm 초과**는 이상치로 볼 수 있어요.  
- 이상치는 **실제 드문 사례**이거나, **측정/입력 오류**일 수 있어 추가 확인이 필요합니다.

4) **모양(왜도)**  
- 왜도가 {skew:.3f} 이면 0에 가까워 **좌우가 비교적 대칭**일 가능성이 있습니다.  
- 왼쪽/오른쪽 끝 구간의 빈도가 더 크면, **왼꼬리/오른꼬리**를 의심해 볼 수 있어요.

5) **결론 예시**  
- "우리 반의 키는 평균 {mean:.1f} cm 정도이며, 중간 50%는 {q1:.1f}~{q3:.1f} cm에 몰려 있다. 이상치는 {out_n}명으로 드물다."
""".format(
    mean=mean, median=median, std=std,
    low1=mean-std, high1=mean+std,
    q1=q1, q3=q3, iqr=iqr,
    lower=lower_bound, upper=upper_bound,
    skew=skewness, out_n=len(outliers)
))

st.subheader("📝 학생 활동")
st.markdown("""
- **활동 1**: 슬라이더로 **구간 개수**를 바꾸며 그래프 모양이 어떻게 달라지는지 비교하세요.  
- **활동 2**: 이상치 기준으로 표시된 학생 수를 확인하고, **이상치가 생기는 이유**를 토론해 보세요.  
- **활동 3**: 평균과 중앙값이 얼마나 다른지 보고, **왜 평균만으로는 분포를 충분히 설명하기 어렵**는지 설명문을 작성해 보세요.
""")
