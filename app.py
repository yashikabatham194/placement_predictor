# ==========================================================
# STUDENT PLACEMENT PREDICTION SYSTEM
# Streamlit deployment — dark navy/black theme
# Run locally with:  streamlit run app.py
#
# Required files in the same folder:
#   - model.pkl                        (trained Pipeline: StandardScaler + RandomForest)
#   - feature_columns.json             (exact column order used at training time)
#   - student_placement_prediction_dataset.csv   (powers the Insights tab, optional)
# ==========================================================

import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Placement Predictor", page_icon="🎯", layout="wide")

# ----------------------------------------------------------
# COLOR TOKENS — dark navy / black, single electric-blue accent
# ----------------------------------------------------------
INK        = "#02040A"
PANEL      = "#0A1020"
PANEL_2    = "#111A30"
LINE       = "#20304F"
TEXT       = "#DCE4F5"
MUTED      = "#7386A8"
ACCENT     = "#2E7CF6"
ACCENT_2   = "#5FA8FF"
GOOD       = "#2FD180"
GOOD_BG    = "#08201A"
BAD        = "#F5A524"
BAD_BG     = "#241708"

st.markdown(f"""
<style>
    .stApp {{ background: {INK} !important; }}
    #MainMenu, header, footer {{ visibility: hidden; }}

    section[data-testid="stAppViewContainer"] * {{
        color: {TEXT};
    }}
    h1, h2, h3, h4 {{ color: {TEXT} !important; font-weight: 700; }}

    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.9rem 1.6rem;
        background: {PANEL};
        border: 1px solid {LINE};
        border-radius: 12px;
        margin-bottom: 1.4rem;
    }}
    .topbar .brand {{
        font-size: 1.15rem; font-weight: 800; color: {TEXT};
        letter-spacing: -0.01em;
    }}
    .topbar .brand span {{ color: {ACCENT}; }}
    .topbar .tag {{
        font-size: 0.72rem; color: {ACCENT_2}; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        border: 1px solid {ACCENT}; border-radius: 999px;
        padding: 0.25rem 0.75rem;
    }}

    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {PANEL} !important;
        border: 1px solid {LINE} !important;
        border-radius: 12px !important;
        padding: 1.4rem 1.6rem !important;
    }}

    div[data-baseweb="input"] input, div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {{
        background: {PANEL_2} !important;
        color: {TEXT} !important;
        border-color: {LINE} !important;
    }}
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {ACCENT} !important;
    }}
    div[role="radiogroup"] label p {{ color: {TEXT} !important; }}

    div.stButton > button, div.stFormSubmitButton > button {{
        background: {ACCENT};
        color: #FFFFFF !important;
        border: none; border-radius: 9px;
        font-weight: 700; width: 100%;
        padding: 0.6em 1em;
    }}
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
        background: {ACCENT_2};
    }}

    div[data-testid="stMetric"] {{
        background: {PANEL_2}; border: 1px solid {LINE};
        border-radius: 10px; padding: 0.7rem 0.9rem;
    }}
    div[data-testid="stMetricValue"] {{ color: {ACCENT_2} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}

    button[data-baseweb="tab"] p {{ color: {MUTED} !important; }}
    button[aria-selected="true"] p {{ color: {ACCENT_2} !important; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}

    .verdict {{
        border-radius: 12px; padding: 1.3rem 1.5rem; margin-top: 0.6rem;
    }}
    .verdict h3 {{ margin: 0 0 0.3rem 0; }}

    .footnote {{
        text-align: center; color: {MUTED}; font-size: 0.8rem;
        margin-top: 2rem; padding-top: 1rem; border-top: 1px solid {LINE};
    }}
</style>
""", unsafe_allow_html=True)


def themed_chart(fig, h=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font_color=TEXT, height=h,
        margin=dict(t=30, b=20, l=10, r=10),
    )
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE)
    return fig


@st.cache_resource
def load_model():
    m = joblib.load("model.pkl")
    cols = json.load(open("feature_columns.json"))
    return m, cols

@st.cache_data
def load_data():
    try:
        return pd.read_csv("student_placement_prediction_dataset.csv")
    except FileNotFoundError:
        return None

try:
    model, FEATURE_COLUMNS = load_model()
    model_ready = True
except FileNotFoundError:
    model, FEATURE_COLUMNS = None, []
    model_ready = False

df = load_data()
data_ready = df is not None


def build_features(ssc, hsc, degree, cgpa, aptitude, comm, tech,
                    internships, projects, certs, backlogs, extracurricular):
    row = {
        "Academic_Avg": (ssc + hsc + degree + cgpa * 10) / 4,
        "Aptitude_Test_Score": aptitude,
        "Backlog_Flag": int(backlogs > 0),
        "Backlogs": backlogs,
        "CGPA": cgpa,
        "Certification_Count": certs,
        "Communication_Score": comm,
        "Degree_Percentage": degree,
        "Experience_Score": internships + projects + certs,
        "Extracurricular_Activities": 1 if extracurricular == "Yes" else 0,
        "HSC_Percentage": hsc,
        "Internships": internships,
        "Projects_Count": projects,
        "SSC_Percentage": ssc,
        "Skill_Index": (aptitude + comm + tech) / 3,
        "Technical_Score": tech,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


st.markdown(f"""
<div class="topbar">
    <div class="brand">🎯 Placement<span>Predictor</span></div>
    <div class="tag">ML Powered</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

with tab_predict:
    if not model_ready:
        st.warning("`model.pkl` / `feature_columns.json` not found next to `app.py` — predictions disabled.")

    left, right = st.columns([1.3, 1])

    with left:
        with st.form("predict_form"):
            st.markdown("#### Student Profile")

            r1c1, r1c2 = st.columns(2)
            ssc = r1c1.slider("SSC %", 35.0, 100.0, 72.0, 0.5)
            hsc = r1c2.slider("HSC %", 35.0, 100.0, 70.0, 0.5)

            r2c1, r2c2 = st.columns(2)
            degree = r2c1.slider("Degree %", 35.0, 100.0, 69.0, 0.5)
            cgpa = r2c2.slider("CGPA (/10)", 4.0, 10.0, 6.9, 0.1)

            r3c1, r3c2, r3c3 = st.columns(3)
            aptitude = r3c1.slider("Aptitude", 20.0, 100.0, 64.0, 1.0)
            comm = r3c2.slider("Communication", 20.0, 100.0, 64.0, 1.0)
            tech = r3c3.slider("Technical", 20.0, 100.0, 70.0, 1.0)

            r4c1, r4c2, r4c3, r4c4 = st.columns(4)
            internships = r4c1.number_input("Internships", 0, 5, 1)
            projects = r4c2.number_input("Projects", 0, 10, 2)
            certs = r4c3.number_input("Certs", 0, 10, 1)
            backlogs = r4c4.number_input("Backlogs", 0, 6, 0)

            extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)
            go = st.form_submit_button("Predict Placement →")

    with right:
        if go:
            if not model_ready:
                st.error("Cannot predict — model files missing.")
            else:
                X = build_features(ssc, hsc, degree, cgpa, aptitude, comm, tech,
                                    internships, projects, certs, backlogs, extracurricular)
                pred = model.predict(X)[0]
                proba = float(model.predict_proba(X)[0][1])

                if pred == 1:
                    st.markdown(f"""
                    <div class="verdict" style="background:{GOOD_BG}; border:1px solid {GOOD};">
                        <h3 style="color:{GOOD} !important;">✅ Likely Placed</h3>
                        <p style="color:{TEXT};">Placement probability: <b style="color:{GOOD};">{proba*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict" style="background:{BAD_BG}; border:1px solid {BAD};">
                        <h3 style="color:{BAD} !important;">⚠️ Likely Not Placed</h3>
                        <p style="color:{TEXT};">Placement probability: <b style="color:{BAD};">{proba*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(proba)

                tips = []
                if backlogs > 0: tips.append("Clear pending backlogs before drives begin.")
                if cgpa < 6.5: tips.append("Push CGPA above the common 6.5 cutoff.")
                if tech < 60: tips.append("Practice DSA and build more technical depth.")
                if comm < 60: tips.append("Run mock interviews to sharpen communication.")
                if internships == 0: tips.append("Take on at least one internship.")
                if projects < 2: tips.append("Add 1–2 more portfolio projects.")
                if tips:
                    st.markdown("##### Suggestions")
                    for t in tips:
                        st.markdown(f"- {t}")
        else:
            st.info("Fill in the form and click **Predict Placement** to see the result here.")

with tab_insights:
    if not data_ready:
        st.warning("`student_placement_prediction_dataset.csv` not found — Insights disabled.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        total = len(df)
        placed = int((df["Placement_Status"] == "Placed").sum())
        k1.metric("Students", f"{total:,}")
        k2.metric("Placed", f"{placed:,}")
        k3.metric("Placement Rate", f"{placed/total*100:.1f}%")
        k4.metric("Avg CGPA", f'{df["CGPA"].mean():.2f}')

        c1, c2 = st.columns(2)
        with c1:
            counts = df["Placement_Status"].value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            fig = px.pie(counts, names="Status", values="Count", hole=0.55,
                         color="Status", color_discrete_map={"Placed": GOOD, "Not Placed": BAD})
            st.plotly_chart(themed_chart(fig), use_container_width=True)
        with c2:
            fig = px.histogram(df, x="CGPA", color="Placement_Status", barmode="overlay",
                                nbins=30, color_discrete_map={"Placed": GOOD, "Not Placed": BAD}, opacity=0.75)
            st.plotly_chart(themed_chart(fig), use_container_width=True)

        spec_rate = (df.groupby("Specialization")["Placement_Status"]
                     .apply(lambda s: (s == "Placed").mean() * 100)
                     .sort_values(ascending=False).reset_index())
        spec_rate.columns = ["Specialization", "Placement Rate (%)"]
        fig = px.bar(spec_rate, x="Specialization", y="Placement Rate (%)",
                     color="Placement Rate (%)", color_continuous_scale="Blues")
        st.plotly_chart(themed_chart(fig, h=380), use_container_width=True)

with tab_about:
    st.markdown("""
    #### About this project
    A machine learning pipeline that predicts whether a student is likely to be placed,
    based on academic scores, skill assessments, and experience indicators — trained
    with a tuned Random Forest inside a `scikit-learn` Pipeline.
    """)
    if FEATURE_COLUMNS:
        st.markdown("**Features used by the model:**")
        st.code(", ".join(FEATURE_COLUMNS), language="text")

st.markdown('<div class="footnote">Student Placement Prediction System</div>', unsafe_allow_html=True)

