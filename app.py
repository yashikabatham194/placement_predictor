"""Premium Student Placement Predictor multi-page Streamlit app."""

import json
import re
from contextlib import contextmanager
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Existing visual language, with the requested premium dark page canvas.
TEAL_DARK = "#0E3B30"
TEAL = "#12594A"
TEAL_SOFT = "#E9F2EF"
GOLD = "#E3A93F"
BG = "#0B0B0B"
CARD = "#FFFFFF"
BORDER = "#E5E8E6"
TEXT = "#1B241F"
MUTED = "#748079"
SUCCESS = "#1E8E5A"
DANGER = "#C4483C"

APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = (
    APP_DIR / "student_placement_prediction_dataset.csv",
    Path(r"C:\Users\hp\Desktop\Office tasks\Project 3\student_placement_prediction_dataset.csv"),
)


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    .stApp {{ background: {BG}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1180px; animation: fadeUp .5s ease; }}
    @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
    .nav-shell {{ display:flex; align-items:center; justify-content:space-between; margin:0 0 24px; padding:4px 2px; }}
    .brand {{ color:#fff; font-size:15px; font-weight:800; letter-spacing:-.2px; }}
    .nav-label {{ color:rgba(255,255,255,.62); font-size:13px; text-align:right; }}
    .hero {{ background:linear-gradient(135deg,{TEAL_DARK} 0%,{TEAL} 100%); border-radius:20px; padding:34px 40px; color:white; margin-bottom:28px; box-shadow:0 10px 30px rgba(14,59,48,.20); }}
    .hero-badge {{ display:none !important; align-items:center; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); padding:5px 14px; border-radius:999px; font-size:12.5px; font-weight:600; letter-spacing:.3px; margin-bottom:14px; }}
    .hero h1 {{ font-size:30px; font-weight:800; margin:0 0 8px; letter-spacing:-.4px; }}
    .hero p {{ font-size:15px; color:rgba(255,255,255,.82); max-width:650px; line-height:1.55; margin:0; }}
    .landing-hero {{ min-height:420px; display:flex; flex-direction:column; justify-content:center; background:linear-gradient(135deg,{TEAL_DARK},#09251E); }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{CARD}; border:1px solid {BORDER} !important; border-radius:16px !important; box-shadow:0 2px 10px rgba(20,40,35,.04); transition:box-shadow .25s ease,transform .25s ease; padding:4px 6px; }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{ box-shadow:0 8px 24px rgba(20,40,35,.09); transform:translateY(-1px); }}
    .card-title {{ font-size:15.5px; font-weight:700; color:{TEAL_DARK}; display:flex; align-items:center; gap:8px; margin-bottom:2px; }}
    .card-subtitle {{ font-size:12.5px; color:{MUTED}; margin-bottom:14px; }}
    [data-testid="stSlider"] div[role="slider"] {{ background-color:{TEAL} !important; box-shadow:0 0 0 4px rgba(18,89,74,.15) !important; }}
    div[data-baseweb="slider"] > div > div {{ background:{TEAL} !important; }}
    [data-testid="stNumberInput"] input, [data-baseweb="select"] > div {{ border-radius:10px !important; border-color:{BORDER} !important; }}
    [data-testid="stRadio"] label, [data-testid="stRadio"] label * {{ color:{TEXT} !important; }}
    [data-testid="stRadio"] label {{ background:{TEAL_SOFT}; padding:6px 14px; border-radius:999px; margin-right:6px; border:1px solid transparent; }}
    .stButton > button, div[data-testid="stFormSubmitButton"] button {{ background:linear-gradient(135deg,{TEAL_DARK},{TEAL}); color:white; border:0; border-radius:12px; padding:11px 18px; font-weight:700; font-size:14px; box-shadow:0 6px 18px rgba(14,59,48,.25); transition:transform .15s ease,box-shadow .15s ease; }}
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {{ transform:translateY(-1px); box-shadow:0 10px 22px rgba(14,59,48,.32); color:white; }}
    .nav-button .stButton > button {{ background:transparent; box-shadow:none; color:rgba(255,255,255,.74); padding:7px 10px; font-size:13px; }}
    .nav-button .stButton > button:hover {{ color:#fff; box-shadow:none; transform:none; }}
    .result-badge {{ display:inline-flex; padding:8px 16px; border-radius:999px; font-weight:700; font-size:14.5px; margin-bottom:6px; }}
    .badge-success {{ background:rgba(30,142,90,.12); color:{SUCCESS}; }} .badge-danger {{ background:rgba(196,72,60,.10); color:{DANGER}; }}
    .gauge-wrap {{ display:flex; flex-direction:column; align-items:center; padding:6px 0 2px; }}
    .gauge-pct {{ font-size:28px; font-weight:800; color:{TEAL_DARK}; }} .gauge-label {{ font-size:12px; color:{MUTED}; font-weight:600; letter-spacing:.4px; text-transform:uppercase; }}
    .result-name {{ color:{TEAL_DARK}; font-size:26px; font-weight:800; margin:0 0 5px; }}
    .result-copy {{ color:{MUTED}; line-height:1.65; font-size:14px; }}
    [data-testid="stMetricValue"] {{ color:{TEAL_DARK}; font-weight:800; }}
    [data-testid="stProgress"] > div > div {{ background:linear-gradient(90deg,{TEAL_DARK},{GOLD}); border-radius:999px; }}
    .footer-chip {{ display:inline-flex; background:{CARD}; border:1px solid {BORDER}; padding:8px 16px; border-radius:999px; font-size:12.5px; color:{MUTED}; margin-top:18px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@contextmanager
def card(title: str, icon: str = "", subtitle: str = ""):
    with st.container(border=True):
        subtitle_markup = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
        st.markdown(
            f'<div style="padding:10px 12px 0"><div class="card-title">{title}</div>{subtitle_markup}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="padding:0 12px 12px">', unsafe_allow_html=True)
        yield
        st.markdown('</div>', unsafe_allow_html=True)


def gauge_svg(pct: float, size: int = 168) -> str:
    pct = max(0.0, min(100.0, pct))
    color = SUCCESS if pct >= 50 else DANGER
    return f'''<div class="gauge-wrap"><div style="width:{size}px;height:{size}px;border-radius:50%;background:conic-gradient({color} {pct * 3.6}deg,{BORDER} 0);display:flex;align-items:center;justify-content:center"><div style="width:{size - 28}px;height:{size - 28}px;border-radius:50%;background:{CARD};display:flex;flex-direction:column;align-items:center;justify-content:center"><span class="gauge-pct">{pct:.1f}%</span><span class="gauge-label">Placement chance</span></div></div></div>'''


@st.cache_resource
def load_artifacts():
    model = joblib.load(APP_DIR / "model.pkl")
    with open(APP_DIR / "feature_columns.json", encoding="utf-8") as file:
        features = json.load(file)
    return model, features


@st.cache_data
def load_students():
    for path in DATA_CANDIDATES:
        if path.exists():
            data = pd.read_csv(path)
            if "Name" in data.columns:
                return data, str(path)
    return pd.DataFrame(), None


def go(page: str):
    st.session_state.page = page


def set_student_values():
    students, _ = load_students()
    name = st.session_state.student_picker
    if not name or students.empty:
        return
    row = students.loc[students["Name"].astype(str) == name].iloc[0]
    mappings = {
        "ssc": "SSC_Percentage", "hsc": "HSC_Percentage", "degree": "Degree_Percentage", "cgpa": "CGPA",
        "aptitude": "Aptitude_Test_Score", "comm": "Communication_Score", "tech": "Technical_Score",
        "internships": "Internships", "projects": "Projects_Count", "certs": "Certification_Count", "backlogs": "Backlogs",
    }
    for key, column in mappings.items():
        if column in row and pd.notna(row[column]):
            st.session_state[key] = float(row[column]) if key in {"ssc", "hsc", "degree", "cgpa", "aptitude", "comm", "tech"} else int(row[column])
    if "Extracurricular_Activities" in row and pd.notna(row["Extracurricular_Activities"]):
        st.session_state.extra = str(row["Extracurricular_Activities"])
    st.session_state.selected_student = name
    st.session_state.selected_salary = row.get("Salary_LPA", row.get("Salary_INR", None))


def canonical_student_name(name: object) -> str:
    """Hide generated numeric suffixes so duplicate-looking names appear once."""
    return re.sub(r"\s+\d+$", "", str(name)).strip()


def load_selected_student_from_dropdown():
    """Map the displayed unique name back to the matching dataset record."""
    selected_name = st.session_state.get("student_dropdown")
    if not selected_name:
        return
    students, _ = load_students()
    raw_names = students["Name"].dropna().astype(str).tolist()
    matching_names = [name for name in raw_names if canonical_student_name(name).casefold() == selected_name.casefold()]
    if not matching_names:
        return
    raw_name = next((name for name in matching_names if name.casefold() == selected_name.casefold()), matching_names[0])
    st.session_state.student_picker = raw_name
    set_student_values()

def run_prediction():
    values = st.session_state
    academic_avg = (values.ssc + values.hsc + values.degree + values.cgpa * 10) / 4
    skill_index = (values.aptitude + values.comm + values.tech) / 3
    input_dict = {
        "Academic_Avg": academic_avg, "Aptitude_Test_Score": values.aptitude,
        "Backlog_Flag": int(values.backlogs > 0), "Backlogs": values.backlogs, "CGPA": values.cgpa,
        "Certification_Count": values.certs, "Communication_Score": values.comm,
        "Degree_Percentage": values.degree, "Experience_Score": values.internships + values.projects + values.certs,
        "Extracurricular_Activities": 1 if values.extra == "Yes" else 0, "HSC_Percentage": values.hsc,
        "Internships": values.internships, "Projects_Count": values.projects, "SSC_Percentage": values.ssc,
        "Skill_Index": skill_index, "Technical_Score": values.tech,
    }
    model, features = load_artifacts()
    frame = pd.DataFrame([input_dict])[features]
    probabilities = model.predict_proba(frame)[0]
    st.session_state.result = {"prediction": int(model.predict(frame)[0]), "placed_prob": float(probabilities[1]), "confidence": float(max(probabilities)), "input": frame}
    go("result")


def navbar():
    a, b, c, d = st.columns([4.2, 1, 1, 1])
    a.markdown('<div class="brand">Student Placement Predictor</div>', unsafe_allow_html=True)
    for column, label, page in ((b, "Home", "home"), (c, "Predict", "predict"), (d, "About", "about")):
        with column:
            st.markdown('<div class="nav-button">', unsafe_allow_html=True)
            if st.button(label, key=f"nav_{label}", use_container_width=True):
                go(page)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def home_page():
    st.markdown('''<div class="hero landing-hero"><div class="hero-badge">AI-Powered Analytics</div><div style="font-size:18px;font-weight:700;margin-bottom:12px;">Welcome!</div><h1>See the next step in every student's placement journey.</h1><p>Use a tuned placement model and a clear academic profile to understand placement likelihood with confidence.</p></div>''', unsafe_allow_html=True)
    if st.button("Predict Placement", key="hero_predict"):
        go("predict")
        st.rerun()


def predict_page():
    st.markdown('''<div class="hero"><div class="hero-badge">AI-Powered Analytics</div><h1>Student Placement Prediction</h1><p>Search for a student to instantly load their profile, then predict placement likelihood with the existing trained model.</p></div>''', unsafe_allow_html=True)
    students, data_path = load_students()
    if students.empty:
        st.warning("Student data was not found. Add student_placement_prediction_dataset.csv beside app.py to enable profile search and auto-fill.")
    else:
        with card("Student Profile", "", "Type to search or scroll through student names"):
            raw_names = students["Name"].dropna().astype(str).tolist()
            display_names = sorted({canonical_student_name(name) for name in raw_names}, key=str.casefold)
            st.selectbox(
                "Enter name",
                display_names,
                index=None,
                placeholder="Type to search or scroll names",
                key="student_dropdown",
                on_change=load_selected_student_from_dropdown,
            )
            if st.session_state.get("selected_student"):
                st.caption(f"Loaded profile for {st.session_state.selected_student}")

    with st.form("placement_form"):
        with card("Academic Details", "", "Board and degree performance"):
            c1, c2, c3 = st.columns(3)
            with c1: st.slider("SSC Percentage", 35.0, 100.0, 72.0, 0.5, key="ssc")
            with c2: st.slider("HSC Percentage", 35.0, 100.0, 70.0, 0.5, key="hsc")
            with c3: st.slider("Degree Percentage", 35.0, 100.0, 69.0, 0.5, key="degree")
            st.slider("CGPA (out of 10)", 4.0, 10.0, 6.9, 0.1, key="cgpa")
        with card("Test & Skill Scores", "", "Aptitude, communication and technical ability"):
            c1, c2, c3 = st.columns(3)
            with c1: st.slider("Aptitude Test Score", 0.0, 100.0, 64.0, 0.5, key="aptitude")
            with c2: st.slider("Communication Score", 0.0, 100.0, 64.0, 0.5, key="comm")
            with c3: st.slider("Technical Score", 0.0, 100.0, 71.0, 0.5, key="tech")
        with card("Experience & Activity", "", "Internships, projects and extracurriculars"):
            c1, c2, c3 = st.columns(3)
            with c1: st.number_input("Internships", 0, 10, 1, key="internships")
            with c2: st.number_input("Projects Count", 0, 15, 2, key="projects")
            with c3: st.number_input("Certification Count", 0, 10, 1, key="certs")
            c1, c2 = st.columns(2)
            with c1: st.number_input("Backlogs", 0, 10, 0, key="backlogs")
            with c2: st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True, key="extra")
        submitted = st.form_submit_button("Predict Placement", use_container_width=True)
    if submitted:
        run_prediction()
        st.rerun()


def result_page():
    result = st.session_state.get("result")
    if not result:
        go("predict")
        st.rerun()
    name = st.session_state.get("selected_student") or "Student"
    placed = result["prediction"] == 1
    probability, confidence = result["placed_prob"] * 100, result["confidence"] * 100
    st.markdown('''<div class="hero"><div class="hero-badge">Prediction Complete</div><h1>Your placement insight is ready</h1><p>A focused view of the model's assessment and the next best steps.</p></div>''', unsafe_allow_html=True)
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        with card("Placement Result", ""):
            st.markdown(f'<div class="result-name">{name}</div>', unsafe_allow_html=True)
            badge = "Likely Placed" if placed else "Placement Improvement Plan"
            css = "badge-success" if placed else "badge-danger"
            st.markdown(f'<div class="result-badge {css}">{badge}</div>', unsafe_allow_html=True)
            message = f"Congratulations, {name}! Based on your academic profile and our model, you have a high probability of getting placed." if placed else "Your profile has clear room to improve. Strengthen technical and aptitude scores, build practical projects or internships, and reduce backlogs to improve placement readiness."
            st.markdown(f'<p class="result-copy">{message}</p>', unsafe_allow_html=True)
        with card("Personalized Next Steps", ""):
            if placed:
                st.write("Keep building interview readiness, tailor your resume to target roles, and continue showcasing projects and certifications.")
            else:
                st.write("Focus on core technical practice, aptitude preparation, communication skills, and completing an internship or portfolio-ready project.")
    with right:
        with card("Model Confidence", "", "Probability and confidence from the current prediction"):
            st.markdown(gauge_svg(probability), unsafe_allow_html=True)
            st.progress(result["placed_prob"])
            c1, c2 = st.columns(2)
            c1.metric("Probability", f"{probability:.1f}%")
            c2.metric("Confidence", f"{confidence:.1f}%")
            salary = st.session_state.get("selected_salary")
            if salary is not None and pd.notna(salary):
                label = "Expected Salary"
                value = f"Rs. {float(salary):.2f} LPA" if "LPA" in str(salary) or float(salary) < 1000 else f"Rs. {float(salary):,.0f}"
                st.metric(label, value)
    if st.button("Predict Another Student", key="again"):
        go("predict")
        st.rerun()


def about_page():
    st.markdown('''<div class="hero"><div class="hero-badge">About the Model</div><h1>Clear placement insights, thoughtfully presented.</h1><p>This tool uses the existing tuned Random Forest model to interpret academic, skill, and experience inputs.</p></div>''', unsafe_allow_html=True)
    with card("How it works", ""):
        st.write("Select a student profile or adjust the values manually, then review the placement probability and model confidence on a dedicated result page.")


if "page" not in st.session_state:
    st.session_state.page = "home"
navbar()
{"home": home_page, "predict": predict_page, "result": result_page, "about": about_page}[st.session_state.page]()


