
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import auth

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"

FEATURES = [
    "attendance_percent",
    "study_hours_per_day",
    "assignment_score",
    "internal_exam1",
    "internal_exam2",
    "class_participation",
    "sleep_hours",
]
TARGET = "final_score"


# ---------------------------------------------------------
# DARK SAAS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1120;
    --panel: #111827;
    --border: #263247;
    --text: #f3f4f6;
    --muted: #9ca3af;
    --green: #34d399;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(11,17,32,.92);
}

[data-testid="stSidebar"] {
    background: #0e1626;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f9fafb !important;
    letter-spacing: -0.035em;
}

p, label, .stMarkdown {
    color: #d1d5db;
}

.small-muted {
    color: var(--muted);
    font-size: .9rem;
}

.hero {
    background:
      radial-gradient(circle at 85% 20%, rgba(16,185,129,.18), transparent 32%),
      linear-gradient(135deg, #172033, #101827 65%);
    border: 1px solid #2b3b50;
    border-radius: 22px;
    padding: 42px;
    margin: 12px 0 24px;
}

.hero h1 {
    font-size: clamp(2.2rem, 5vw, 4rem);
    line-height: 1.08;
    margin-bottom: 12px;
}

.eyebrow {
    color: #34d399;
    text-transform: uppercase;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .14em;
}

.metric-card {
    background: linear-gradient(145deg, #151f31, #111827);
    border: 1px solid #29364b;
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 132px;
}

.metric-label {
    color: #9ca3af;
    font-size: .86rem;
    margin-bottom: 10px;
}

.metric-value {
    color: #f9fafb;
    font-size: clamp(1.5rem, 3vw, 2.1rem);
    font-weight: 700;
    line-height: 1.2;
}

.metric-note {
    color: #34d399;
    font-size: .8rem;
    margin-top: 8px;
}

.panel {
    background: #111827;
    border: 1px solid #29364b;
    border-radius: 16px;
    padding: 22px;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
    border: 1px solid #334155;
}

div.stButton > button[kind="primary"] {
    background: #10b981;
    color: #06251d;
    border: 1px solid #10b981;
}

div.stButton > button[kind="primary"]:hover {
    background: #34d399;
    border-color: #34d399;
    color: #06251d;
}

div[data-testid="stForm"] {
    background: #111827;
    border: 1px solid #29364b;
    padding: 20px;
    border-radius: 16px;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #29364b;
    padding: 16px;
    border-radius: 14px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29364b;
    border-radius: 12px;
}

hr {
    border-color: #263247;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }
    .hero {
        padding: 24px 18px;
        border-radius: 16px;
    }
    .metric-card {
        padding: 15px;
        min-height: 110px;
    }
    h1 {
        font-size: 2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE + HELPERS
# ---------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"


def go(page):
    st.session_state.page = page
    st.rerun()


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <h1 style="margin-top:6px">{title}</h1>
        <p class="small-muted">{description}</p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
def home_page():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Student intelligence platform</div>
      <h1>Understand performance.<br>
      <span style="color:#34d399">Support potential.</span></h1>
      <p style="max-width:680px;color:#cbd5e1;font-size:1.08rem">
        Explore academic patterns and estimate student outcomes
        with an interactive machine-learning demonstration.
      </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("A clearer view of student progress")
        st.write(
            "EduPredict AI brings academic inputs and predictive "
            "analytics into one simple workspace. Explore a sample "
            "dataset, test different inputs, and learn how a model "
            "generates an estimate."
        )

        if st.button(
            "Get started →",
            type="primary",
            use_container_width=True,
        ):
            go("Login")

    with right:
        st.markdown("""
        <div class="panel">
          <div class="eyebrow">Inside EduPredict</div>
          <h3>One workspace, useful insights</h3>
          <p>✦ Student score estimation</p>
          <p>✦ Performance distribution</p>
          <p>✦ Academic input exploration</p>
          <p>✦ Model and methodology overview</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)

    with a:
        metric_card("PREDICTION", "Score estimate", "Regression model")
    with b:
        metric_card("EXPLORATION", "Interactive", "Change inputs and compare")
    with c:
        metric_card("LEARNING", "Explainable", "Review model limitations")

    st.caption(
        "Educational demonstration using a sample dataset. "
        "Not intended for high-stakes decisions about students."
    )


# ---------------------------------------------------------
# LOGIN + SIGNUP PAGE
# ---------------------------------------------------------

def login_page():
    # Initialize login/signup view
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    # Center the authentication form
    _, center, _ = st.columns([1, 1.15, 1])

    with center:
        st.markdown(
            '<div class="eyebrow">EDUPREDICT AI</div>',
            unsafe_allow_html=True,
        )

        st.title("Welcome to EduPredict AI")
        st.write("Sign in or create your student analytics account.")

        # Switch between login and registration
        view = st.radio(
            "Account",
            ["Sign in", "Create account"],
            horizontal=True,
            label_visibility="collapsed",
            index=(
                0 if st.session_state.auth_view == "login" else 1
            ),
        )

        st.session_state.auth_view = (
            "login" if view == "Sign in" else "signup"
        )

        # Show registration success once
        if "signup_success" in st.session_state:
            st.success(st.session_state.pop("signup_success"))

        # -----------------------------------------
        # SIGN IN
        # -----------------------------------------
        if st.session_state.auth_view == "login":
            st.subheader("Sign in")

            with st.form("login_form"):
                email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )

                submitted = st.form_submit_button(
                    "Sign in",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if not email.strip() or not password:
                    st.error(
                        "Please enter your email address and password."
                    )
                else:
                    user = auth.login_user(
                        email.strip(),
                        password,
                    )

                    if user:
                        st.session_state.logged_in = True
                        st.session_state.current_user = user
                        st.session_state.page = "Overview"
                        st.rerun()
                    else:
                        st.error(
                            "Incorrect email or password. "
                            "Please check your details or create an account."
                        )

            st.caption(
                "Forgot your password? Email recovery is not set up yet."
            )

            st.write("")
            st.write("Don't have an account? Select Create account above.")

        # -----------------------------------------
        # CREATE ACCOUNT
        # -----------------------------------------
        else:
            st.subheader("Create your account")

            with st.form("signup_form"):
                full_name = st.text_input(
                    "Full name",
                    placeholder="Enter your full name",
                )

                email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                )

                username = st.text_input(
                    "Username",
                    placeholder="Choose a username",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="At least 8 characters",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Re-enter your password",
                )

                submitted = st.form_submit_button(
                    "Create account",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                # Validate required fields
                if not all([
                    full_name.strip(),
                    email.strip(),
                    username.strip(),
                    password,
                    confirm_password,
                ]):
                    st.error("Please fill in all fields.")

                elif "@" not in email or "." not in email.split("@")[-1]:
                    st.error("Please enter a valid email address.")

                elif len(username.strip()) < 3:
                    st.error(
                        "Username must contain at least 3 characters."
                    )

                elif len(password) < 8:
                    st.error(
                        "Password must contain at least 8 characters."
                    )

                elif password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    success, message = auth.register_user(
                        full_name.strip(),
                        email.strip(),
                        username.strip(),
                        password,
                    )

                    if success:
                        st.session_state.signup_success = (
                            "Account created successfully! "
                            "Please log in with your email and password."
                        )
                        st.session_state.auth_view = "login"
                        st.rerun()
                    else:
                        st.error(message)

    # -----------------------------------------
    # BACK TO HOME
    # -----------------------------------------
    st.write("")

    if st.button("← Back to home"):
        st.session_state.auth_view = "login"
        go("Home")

# ---------------------------------------------------------
# DATA + MODEL
# ---------------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    return df


@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    reg = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=2,
    )
    reg.fit(X_train, y_train)

    predicted = reg.predict(X_test)
    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)

    clf = None
    accuracy = None

    if "result" in data.columns:
        labels = data["result"].fillna("").astype(str).str.strip()
        valid = labels.ne("")

        if labels[valid].nunique() > 1:
            Xc = data.loc[valid, FEATURES]
            yc = labels[valid]

            stratify_labels = (
                yc if yc.value_counts().min() >= 2 else None
            )

            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                Xc,
                yc,
                test_size=0.2,
                random_state=42,
                stratify=stratify_labels,
            )

            clf = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                min_samples_leaf=2,
                class_weight="balanced",
            )
            clf.fit(Xc_train, yc_train)
            accuracy = accuracy_score(
                yc_test,
                clf.predict(Xc_test),
            )

    return reg, clf, mae, r2, accuracy


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 18px">
          <div style="font-size:1.25rem;font-weight:700;color:#f9fafb">
            🎓 EduPredict <span style="color:#34d399">AI</span>
          </div>
          <div style="font-size:.68rem;letter-spacing:.12em;color:#9ca3af">
            STUDENT INTELLIGENCE
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("WORKSPACE")
        pages = [
            "Overview",
            "Predict student",
            "Insights",
            "Model information",
        ]

        current = st.session_state.page
        selected = st.radio(
            "Navigation",
            pages,
            index=pages.index(current) if current in pages else 0,
            label_visibility="collapsed",
        )

        if selected != current:
            st.session_state.page = selected
            st.rerun()

        st.divider()
        st.caption("ACCOUNT")

        user = st.session_state.current_user or {}
        display_name = user.get("full_name") or user.get("username") or "User"
        st.write(f"👤 {display_name}")

        if st.button("Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.page = "Home"
            st.session_state.last_prediction = None
            st.rerun()


# ---------------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------------
def overview_page(df, mae):
    top_heading(
        "WORKSPACE / OVERVIEW",
        "Performance overview",
        "A snapshot of the sample dataset and your latest prediction.",
    )

    latest = st.session_state.last_prediction

    if latest:
        score = latest["score"]
        outcome = latest["outcome"]
        note = "Your most recent input"
    else:
        score = float(df[TARGET].mean())
        outcome = "Dataset average"
        note = "Average score in sample data"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("ESTIMATED SCORE", f"{score:.1f}/100", note)
    with c2:
        metric_card(
            "ESTIMATED OUTCOME",
            outcome,
            "Based on latest input" if latest else "Dataset snapshot",
        )
    with c3:
        metric_card("STUDENTS IN DATA", f"{len(df):,}", "Usable sample records")
    with c4:
        metric_card("MODEL MAE", f"{mae:.2f}", "Test-set score points")

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Score distribution")
        st.caption("How final scores are distributed in the sample data.")
        st.bar_chart(
            df[TARGET].round(-1).value_counts().sort_index(),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Average by attendance")
        st.caption("Grouped averages; descriptive, not causal.")

        bins = [0, 60, 75, 90, 101]
        labels = ["0–60%", "61–75%", "76–90%", "91–100%"]
        temp = df.copy()

        temp["attendance_band"] = pd.cut(
            temp["attendance_percent"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        chart = temp.groupby(
            "attendance_band",
            observed=False,
        )[TARGET].mean().dropna()

        st.line_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Recent sample records")
    display_cols = FEATURES + [TARGET]
    st.dataframe(
        df[display_cols].head(8),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
def predict_page(reg, clf):
    top_heading(
        "WORKSPACE / PREDICTION",
        "Predict student performance",
        "Enter academic and study-habit inputs to generate an estimate.",
    )

    st.info(
        "Use fictional or sample values for this demonstration. "
        "Predictions are estimates and may be inaccurate."
    )

    with st.form("prediction_form"):
        st.subheader("Student details")
        c1, c2 = st.columns(2)

        with c1:
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            study = st.slider("Study hours per day", 0.0, 16.0, 3.0, 0.5)
            assignment = st.slider("Assignment score (%)", 0, 100, 75)
            participation = st.slider(
                "Class participation (1–5)",
                1,
                5,
                3,
            )

        with c2:
            exam1 = st.slider("Internal exam 1 (%)", 0, 100, 70)
            exam2 = st.slider("Internal exam 2 (%)", 0, 100, 72)
            sleep = st.slider("Sleep hours per night", 0.0, 16.0, 8.0, 0.5)

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        row = pd.DataFrame([{
            "attendance_percent": attendance,
            "study_hours_per_day": study,
            "assignment_score": assignment,
            "internal_exam1": exam1,
            "internal_exam2": exam2,
            "class_participation": participation,
            "sleep_hours": sleep,
        }])[FEATURES]

        score = float(np.clip(reg.predict(row)[0], 0, 100))

        if clf is not None:
            outcome = str(clf.predict(row)[0])
        else:
            outcome = "Pass" if score >= 50 else "Fail"

        st.session_state.last_prediction = {
            "score": score,
            "outcome": outcome,
            "inputs": row.iloc[0].to_dict(),
        }

    latest = st.session_state.last_prediction

    if latest:
        st.write("")
        st.subheader("Prediction result")
        a, b = st.columns(2)

        with a:
            metric_card(
                "ESTIMATED FINAL SCORE",
                f"{latest['score']:.1f}/100",
                "Model estimate",
            )

        with b:
            metric_card(
                "ESTIMATED OUTCOME",
                latest["outcome"],
                "Model classification / score threshold",
            )

        st.progress(int(round(latest["score"])))
        st.caption(
            "The estimate is based on patterns learned from the sample data. "
            "It is not a guarantee of a student's actual result."
        )

        st.subheader("Your submitted inputs")
        st.dataframe(
            pd.DataFrame([latest["inputs"]]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown(
            '<div class="panel">Your prediction will appear here after '
            'you submit the student details.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# INSIGHTS PAGE
# ---------------------------------------------------------
def insights_page(df):
    top_heading(
        "WORKSPACE / INSIGHTS",
        "Student performance insights",
        "Explore relationships and distributions in the available sample data.",
    )

    selected = st.selectbox(
        "Compare final score against",
        FEATURES,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Input vs final score")
        st.caption("Each point is a sample record.")
        st.scatter_chart(
            df,
            x=selected,
            y=TARGET,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Average final score")
        st.caption("Mean score for selected input groups.")

        if selected in [
            "attendance_percent",
            "assignment_score",
            "internal_exam1",
            "internal_exam2",
        ]:
            bins = [0, 50, 60, 70, 80, 90, 101]
            temp = df.copy()
            temp["group"] = pd.cut(
                temp[selected],
                bins=bins,
                include_lowest=True,
            )
            means = temp.groupby(
                "group",
                observed=False,
            )[TARGET].mean().dropna()
            means.index = means.index.astype(str)
        else:
            means = df.groupby(selected)[TARGET].mean().sort_index()

        st.bar_chart(means, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Correlation with final score")

    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET]
    corr = corr.drop(TARGET).sort_values()

    st.bar_chart(corr)
    st.caption(
        "Correlation describes association in this dataset. "
        "It does not establish that an input causes a score to change."
    )


# ---------------------------------------------------------
# MODEL INFORMATION PAGE
# ---------------------------------------------------------
def model_page(mae, r2, accuracy):
    top_heading(
        "WORKSPACE / MODEL",
        "How EduPredict AI works",
        "A simple overview of the model, inputs, evaluation and limitations.",
    )

    st.subheader("Model architecture")
    st.markdown("""
    <div class="panel">
      <b>1. Input data</b><br>
      Attendance, study hours, assignment score, internal exams,
      participation and sleep hours.
      <br><br>
      <b>2. Data preparation</b><br>
      Load the CSV, convert the required columns to numeric values,
      and remove rows with missing required values.
      <br><br>
      <b>3. Machine-learning models</b><br>
      Random Forest Regression estimates final score. If the dataset
      includes multiple outcome labels, a Random Forest Classifier
      also predicts the result label.
      <br><br>
      <b>4. Output</b><br>
      Estimated final score and, when available, predicted outcome.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Evaluation")
    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("MAE", f"{mae:.2f}", "Average absolute error on test split")

    with c2:
        metric_card("R²", f"{r2:.3f}", "Regression test-set metric")

    with c3:
        metric_card(
            "CLASSIFIER ACCURACY",
            f"{accuracy:.1%}" if accuracy is not None else "N/A",
            "Outcome test-set accuracy"
            if accuracy is not None
            else "No usable result labels",
        )

    st.write("")
    st.subheader("Important limitations")
    st.markdown("""
    - The project uses a sample dataset and its patterns may not represent real students.
    - Model metrics depend on the data and the random train/test split.
    - A high test score does not prove the model is fair or reliable in another setting.
    - Predictions should not determine admissions, grades, scholarships,
      discipline or access to educational opportunities.
    - This account system is a local project feature, not production-grade authentication.
    """)


# ---------------------------------------------------------
# APP ROUTER
# ---------------------------------------------------------

# Show public pages before loading the dataset.
if not st.session_state.logged_in:
    if st.session_state.page == "Login":
        login_page()
    else:
        home_page()
    st.stop()


# Load data only after the user has signed in.
try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if data is None or data.empty:
    st.error(
        "Dataset not found or empty. Expected file: "
        "data/student_performance.csv"
    )
    st.stop()

if len(data) < 5:
    st.error("The dataset needs at least 5 usable rows to train the model.")
    st.stop()

try:
    regressor, classifier, model_mae, model_r2, model_accuracy = train_models(data)
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()


sidebar()
page = st.session_state.page

if page == "Overview":
    overview_page(data, model_mae)

elif page == "Predict student":
    predict_page(regressor, classifier)

elif page == "Insights":
    insights_page(data)

elif page == "Model information":
    model_page(model_mae, model_r2, model_accuracy)

else:
    go("Overview")