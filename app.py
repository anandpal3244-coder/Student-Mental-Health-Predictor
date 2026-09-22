import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mental Health Score Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# OPTIONAL LOTTIE
# =========================================================

try:
    import requests
    from streamlit_lottie import st_lottie

    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Mental_Health_Model.pkl"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_COLUMNS = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "group_country",
    "Stress_Level",
]


GENDER_OPTS = [
    "Female",
    "Male",
]

ACADEMIC_OPTS = [
    "High School",
    "Undergraduate",
    "Graduate",
]

PLATFORM_OPTS = [
    "Facebook",
    "Instagram",
    "KakaoTalk",
    "LINE",
    "LinkedIn",
    "Snapchat",
    "TikTok",
    "Twitter",
    "VKontakte",
    "WeChat",
    "WhatsApp",
    "YouTube",
]

PURPOSE_OPTS = [
    "Education",
    "Entertainment",
    "Networking",
    "News",
]

COUNTRY_OPTS = [
    "Australia",
    "Canada",
    "France",
    "Germany",
    "India",
    "Mexico",
    "Other",
    "Turkey",
    "UK",
    "USA",
]

STRESS_OPTS = [
    "Low",
    "Medium",
    "High",
    "Very High",
]


# =========================================================
# MODEL METRICS
# =========================================================

MODEL_METRICS = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest (default)",
            "Random Forest (tuned)",
        ],
        "R2 (Test)": [
            0.7398,
            0.8773,
            0.8655,
        ],
        "R2 (Train)": [
            0.7237,
            0.9808,
            0.9549,
        ],
        "MAE": [
            0.5362,
            0.3483,
            0.3690,
        ],
        "RMSE": [
            0.6760,
            0.4643,
            0.4860,
        ],
    }
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(
            120deg,
            #0f0c29,
            #302b63,
            #24243e,
            #1b1035
        );

        background-size: 300% 300%;
        animation: bgShift 18s ease infinite;
    }

    @keyframes bgShift {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    /* Hero title */
    .hero-title {

        font-size: 3rem;
        font-weight: 800;
        text-align: center;

        background: linear-gradient(
            90deg,
            #38f9d7,
            #43e97b,
            #f9d423,
            #ff5f6d,
            #a1c4fd,
            #38f9d7
        );

        background-size: 400% 400%;

        -webkit-background-clip: text;
        background-clip: text;

        color: transparent;

        animation: gradientMove 8s ease infinite;

        margin-bottom: 0;
    }

    @keyframes gradientMove {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    /* Subtitle */
    .hero-subtitle {

        text-align: center;

        color: #cfd3ff;

        font-size: 1.05rem;

        font-weight: 300;

        margin-top: 0.2rem;

        margin-bottom: 1.6rem;

        opacity: 0.9;
    }


    /* Floating emoji */
    .floaty {

        display: inline-block;

        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {

        0% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-10px);
        }

        100% {
            transform: translateY(0px);
        }

    }


    /* Fade in */
    .fade-in {

        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {

        from {
            opacity: 0;
            transform: translateY(12px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }


    /* Glass cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {

        background: rgba(255, 255, 255, 0.06);

        border-radius: 18px !important;

        border: 1px solid rgba(255, 255, 255, 0.14) !important;

        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);

        backdrop-filter: blur(10px);

        padding: 4px 6px;

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {

        transform: translateY(-3px);

        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.35);
    }


    /* Section header */
    .section-header {

        font-size: 1.4rem;

        font-weight: 700;

        color: #f2f2ff;

        border-left: 5px solid #43e97b;

        padding-left: 12px;

        margin: 0.4rem 0 1rem 0;
    }


    /* Buttons */
    .stButton > button {

        background: linear-gradient(
            90deg,
            #43e97b,
            #38f9d7
        );

        color: #0f0c29;

        font-weight: 700;

        border: none;

        border-radius: 30px;

        padding: 0.65rem 2rem;

        font-size: 1.05rem;

        transition: all 0.25s ease;

        box-shadow:
            0 4px 18px rgba(67, 233, 123, 0.35);
    }

    .stButton > button:hover {

        transform: scale(1.045);

        box-shadow:
            0 8px 26px rgba(67, 233, 123, 0.55);

        color: #0f0c29;
    }


    /* Badge */
    .badge {

        display: inline-block;

        padding: 0.35rem 1rem;

        border-radius: 999px;

        font-weight: 700;

        font-size: 0.95rem;
    }


    /* Sidebar */
    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #1b1035,
            #0f0c29
        );

        border-right:
            1px solid rgba(255,255,255,0.08);
    }


    /* Tabs */
    button[data-baseweb="tab"] {

        font-size: 1.02rem;

        font-weight: 600;

        color: #cfd3ff;
    }

    button[data-baseweb="tab"][aria-selected="true"] {

        color: #43e97b;
    }


    /* Hide Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Text */
    p,
    li,
    label,
    .stMarkdown,
    span {

        color: #e8e8ff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path):
    """
    Load the trained model only once.
    """

    return joblib.load(path)


def load_lottie_url(url, timeout=4):
    """
    Load optional Lottie animation.
    Returns None if unavailable.
    """

    if not LOTTIE_AVAILABLE:
        return None

    try:

        response = requests.get(
            url,
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json()

    except Exception:
        pass

    return None


def score_category(score):
    """
    Convert numerical score into a descriptive category.
    """

    if score < 4.5:

        return (
            "Needs Attention",
            "#ff5f6d"
        )

    elif score < 6.0:

        return (
            "Fair",
            "#f9d423"
        )

    elif score < 7.5:

        return (
            "Good",
            "#43e97b"
        )

    else:

        return (
            "Excellent",
            "#38f9d7"
        )


def make_gauge(score):
    """
    Create Plotly gauge chart.
    """

    label, color = score_category(score)

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=score,

            number={
                "suffix": " / 10",
                "font": {
                    "size": 42,
                    "color": color
                }
            },

            title={
                "text": (
                    "Predicted Mental Health Score"
                    f"<br><span style='font-size:0.8em;"
                    f"color:{color}'>{label}</span>"
                ),
                "font": {
                    "size": 18,
                    "color": "#e8e8ff"
                }
            },

            gauge={

                "axis": {
                    "range": [0, 10],
                    "tickcolor": "#e8e8ff",
                    "tickfont": {
                        "color": "#e8e8ff"
                    }
                },

                "bar": {
                    "color": color,
                    "thickness": 0.28
                },

                "bgcolor": "rgba(0,0,0,0)",

                "borderwidth": 0,

                "steps": [

                    {
                        "range": [0, 4.5],
                        "color":
                            "rgba(255,95,109,0.25)"
                    },

                    {
                        "range": [4.5, 6.0],
                        "color":
                            "rgba(249,212,35,0.25)"
                    },

                    {
                        "range": [6.0, 7.5],
                        "color":
                            "rgba(67,233,123,0.25)"
                    },

                    {
                        "range": [7.5, 10],
                        "color":
                            "rgba(56,249,215,0.25)"
                    },
                ],

                "threshold": {

                    "line": {
                        "color": "white",
                        "width": 3
                    },

                    "thickness": 0.85,

                    "value": score,
                },
            },
        )
    )

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        font={
            "color": "#e8e8ff"
        },

        height=340,

        margin={
            "l": 25,
            "r": 25,
            "t": 70,
            "b": 10
        },
    )

    return fig


def predict_score(model, inputs):
    """
    Prepare user input and make prediction.
    """

    row = pd.DataFrame(
        [inputs],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(row)[0]

    return float(
        np.clip(prediction, 0, 10)
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#43e97b;'>🧠 Project Menu</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        **📌 About this app**

        Predicts a student's **Mental Health Score (0–10)**
        from social media habits, sleep, study and
        physical activity patterns using a trained
        **Random Forest regression pipeline**.
        """
    )

    st.markdown("---")

    st.markdown("**🛠️ Tech Stack**")

    st.markdown(
        "`Python` · `Scikit-learn` · `Pandas` · "
        "`Streamlit` · `Plotly`"
    )

    st.markdown("---")

    st.markdown("**👤 Author**")

    st.markdown(
        "Anand Kumar — *Data Analyst*"
    )

    st.markdown(
        "[GitHub](https://github.com/anandpal3244-coder) · "
        "[LinkedIn](https://www.linkedin.com/in/anand-pal-6a657b393/)"
    )

    st.markdown(
        "📧 anandpal3244@gmail.com"
    )

    st.markdown("---")

    st.caption(
        "Model: Mental_Health_Model.pkl"
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero-title fade-in">
        <span class="floaty">🧠</span>
        Mental Health Score Predictor
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle fade-in">
        Understanding how social media & lifestyle habits
        relate to student well-being
    </div>
    """,
    unsafe_allow_html=True
)


# Optional animation
lottie_brain = load_lottie_url(
    "https://assets9.lottiefiles.com/packages/lf20_puciaact.json"
)

if lottie_brain is not None:

    st_lottie(
        lottie_brain,
        height=180,
        key="hero_lottie"
    )


st.markdown("---")


# =========================================================
# MODEL FILE CHECK
# =========================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "⚠️ Could not find "
        "`Mental_Health_Model.pkl`.\n\n"
        f"Expected location:\n\n"
        f"`{MODEL_PATH}`\n\n"
        "Place the model file in the same folder as "
        "`app.py` and restart the application."
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model(MODEL_PATH)

except Exception as error:

    st.error(
        "⚠️ Failed to load the trained model."
    )

    st.exception(error)

    st.stop()


# =========================================================
# TABS
# =========================================================

tab_predict, tab_insights, tab_model, tab_about = st.tabs(
    [
        "🔮 Predict",
        "📊 Dataset Insights",
        "🤖 Model Performance",
        "ℹ️ About Project",
    ]
)


# =========================================================
# TAB 1 — PREDICT
# =========================================================

with tab_predict:

    st.markdown(
        "<div class='section-header'>"
        "Enter Student Details"
        "</div>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.slider(
                "🎂 Age",
                min_value=18,
                max_value=24,
                value=21
            )

            gender = st.selectbox(
                "🚻 Gender",
                GENDER_OPTS
            )

            academic_level = st.selectbox(
                "🎓 Academic Level",
                ACADEMIC_OPTS,
                index=1
            )

        with c2:

            platform = st.selectbox(
                "📱 Most Used Platform",
                PLATFORM_OPTS,
                index=1
            )

            purpose = st.selectbox(
                "🎯 Purpose of Use",
                PURPOSE_OPTS,
                index=1
            )

            country = st.selectbox(
                "🌍 Country",
                COUNTRY_OPTS,
                index=9
            )

        with c3:

            stress = st.select_slider(
                "😖 Stress Level",
                options=STRESS_OPTS,
                value="Medium"
            )

            usage_hours = st.slider(
                "⏱️ Avg Daily Social Media Usage (hrs)",
                min_value=1.0,
                max_value=8.8,
                value=5.0,
                step=0.1
            )

            unlocks = st.slider(
                "🔓 Daily Phone Unlocks",
                min_value=62,
                max_value=273,
                value=171
            )


    with st.container(border=True):

        c4, c5, c6 = st.columns(3)

        with c4:

            study_hours = st.slider(
                "📚 Study Hours / day",
                min_value=0.3,
                max_value=8.3,
                value=3.0,
                step=0.1
            )

        with c5:

            activity_hours = st.slider(
                "🏃 Physical Activity (hrs/day)",
                min_value=0.0,
                max_value=4.1,
                value=1.8,
                step=0.1
            )

        with c6:

            sleep_hours = st.slider(
                "😴 Sleep (hrs/night)",
                min_value=3.6,
                max_value=9.9,
                value=6.6,
                step=0.1
            )


    st.write("")

    predict_clicked = st.button(
        "✨ Predict Mental Health Score",
        use_container_width=True
    )


    if predict_clicked:

        with st.spinner(
            "Analyzing lifestyle patterns..."
        ):

            inputs = {

                "Study_Hours":
                    study_hours,

                "Age":
                    age,

                "Avg_Daily_Usage_Hours":
                    usage_hours,

                "Daily_Unlocks":
                    unlocks,

                "Physical_Activity_Hours":
                    activity_hours,

                "Sleep_Hours_Per_Night":
                    sleep_hours,

                "Gender":
                    gender,

                "Academic_Level":
                    academic_level,

                "Most_Used_Platform":
                    platform,

                "Purpose_Of_Use":
                    purpose,

                "group_country":
                    country,

                "Stress_Level":
                    stress,
            }


            try:

                score = predict_score(
                    model,
                    inputs
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)

                st.stop()


        label, color = score_category(score)

        st.balloons()


        r1, r2 = st.columns([1.1, 1])


        with r1:

            with st.container(border=True):

                st.plotly_chart(
                    make_gauge(score),
                    use_container_width=True
                )


        with r2:

            with st.container(border=True):

                st.markdown(
                    "<div class='section-header'>"
                    "Interpretation"
                    "</div>",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
                    <span
                        class="badge"
                        style="
                            background:{color}22;
                            color:{color};
                            border:1px solid {color};
                        "
                    >
                        {label}
                    </span>
                    """,
                    unsafe_allow_html=True
                )


                st.write("")


                st.metric(
                    "Predicted Score",
                    f"{score:.2f} / 10"
                )


                tips = {

                    "Needs Attention":
                        "Consider reducing screen time, "
                        "prioritizing sleep and maintaining "
                        "regular physical activity. "
                        "Talking with someone you trust may "
                        "also be helpful.",

                    "Fair":
                        "A few lifestyle adjustments such as "
                        "more sleep, structured study breaks "
                        "or reducing late-night scrolling "
                        "may support better well-being.",

                    "Good":
                        "Your selected habits appear "
                        "reasonably balanced. Continue "
                        "monitoring stress and screen time, "
                        "especially during exams.",

                    "Excellent":
                        "The selected sleep, study, activity "
                        "and social-media patterns are "
                        "associated with a high predicted score."
                }


                st.info(
                    tips[label]
                )


                st.markdown(
                    "**Your inputs vs. dataset average**"
                )


                compare_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Usage Hrs",
                            "Sleep Hrs",
                            "Study Hrs",
                            "Activity Hrs",
                        ],

                        "You": [
                            usage_hours,
                            sleep_hours,
                            study_hours,
                            activity_hours,
                        ],

                        "Dataset Avg": [
                            5.08,
                            6.63,
                            3.01,
                            1.75,
                        ],
                    }
                )


                fig_cmp = px.bar(
                    compare_df,
                    x="Metric",
                    y=["You", "Dataset Avg"],
                    barmode="group",
                    color_discrete_sequence=[
                        "#43e97b",
                        "#8a8fd9",
                    ],
                )


                fig_cmp.update_layout(

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(0,0,0,0)",

                    font={
                        "color": "#e8e8ff"
                    },

                    height=280,

                    legend_title_text="",

                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10,
                    ),
                )


                st.plotly_chart(
                    fig_cmp,
                    use_container_width=True
                )


    else:

        st.info(
            "👆 Fill in the details above and click "
            "**Predict Mental Health Score** to see "
            "the prediction."
        )


# =========================================================
# TAB 2 — DATASET INSIGHTS
# =========================================================

with tab_insights:

    st.markdown(
        "<div class='section-header'>"
        "Explore the Dataset"
        "</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        Upload the original
        **Student Social Media And Mental Health Impact.csv**
        to generate live interactive charts.

        This upload is optional. The predictor works
        without the dataset.
        """
    )


    uploaded = st.file_uploader(
        "Upload dataset CSV",
        type=["csv"]
    )


    if uploaded is not None:

        try:

            df = pd.read_csv(uploaded)

            st.success(
                f"Loaded {df.shape[0]:,} rows × "
                f"{df.shape[1]} columns"
            )


            # -------------------------------------------------
            # Distribution + Stress
            # -------------------------------------------------

            with st.container(border=True):

                cA, cB = st.columns(2)


                with cA:

                    if "Mental_Health_Score" in df.columns:

                        fig = px.histogram(
                            df,
                            x="Mental_Health_Score",
                            nbins=25,
                            title=(
                                "Distribution of "
                                "Mental Health Score"
                            ),
                            color_discrete_sequence=[
                                "#43e97b"
                            ],
                        )


                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={
                                "color": "#e8e8ff"
                            },
                        )


                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )


                with cB:

                    required = {
                        "Stress_Level",
                        "Mental_Health_Score",
                    }


                    if required.issubset(df.columns):

                        stress_order = [
                            "Low",
                            "Medium",
                            "High",
                            "Very High",
                        ]


                        fig = px.box(
                            df,
                            x="Stress_Level",
                            y="Mental_Health_Score",
                            category_orders={
                                "Stress_Level":
                                    stress_order
                            },
                            color="Stress_Level",
                            title=(
                                "Stress Level vs "
                                "Mental Health Score"
                            ),
                        )


                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={
                                "color": "#e8e8ff"
                            },
                            showlegend=False,
                        )


                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )


            # -------------------------------------------------
            # Usage + Platform
            # -------------------------------------------------

            with st.container(border=True):

                cC, cD = st.columns(2)


                with cC:

                    required = {
                        "Avg_Daily_Usage_Hours",
                        "Mental_Health_Score",
                    }


                    if required.issubset(df.columns):

                        color_col = (
                            "Stress_Level"
                            if "Stress_Level" in df.columns
                            else None
                        )


                        fig = px.scatter(
                            df,
                            x="Avg_Daily_Usage_Hours",
                            y="Mental_Health_Score",
                            color=color_col,
                            opacity=0.7,
                            title=(
                                "Usage Hours vs "
                                "Mental Health Score"
                            ),
                        )


                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={
                                "color": "#e8e8ff"
                            },
                        )


                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )


                with cD:

                    if "Most_Used_Platform" in df.columns:

                        counts = (
                            df["Most_Used_Platform"]
                            .value_counts()
                            .reset_index()
                        )


                        counts.columns = [
                            "Platform",
                            "Count",
                        ]


                        fig = px.bar(
                            counts,
                            x="Platform",
                            y="Count",
                            color="Platform",
                            title="Most Used Platform",
                        )


                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={
                                "color": "#e8e8ff"
                            },
                            showlegend=False,
                        )


                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )


            # -------------------------------------------------
            # Correlation
            # -------------------------------------------------

            with st.container(border=True):

                numeric_df = (
                    df.select_dtypes(
                        include="number"
                    )
                )


                if not numeric_df.empty:

                    correlation = (
                        numeric_df.corr(
                            numeric_only=True
                        )
                    )


                    fig = px.imshow(
                        correlation,
                        text_auto=".2f",
                        color_continuous_scale="Tealgrn",
                        title="Correlation Heatmap",
                    )


                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={
                            "color": "#e8e8ff"
                        },
                    )


                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


        except Exception as error:

            st.error(
                "Could not process the uploaded CSV."
            )

            st.exception(error)


    else:

        st.markdown(
            "#### Dataset Snapshot "
            "(from training notebook)"
        )


        st.markdown(
            """
            - **Rows:** 5,000 students
            - **Features:** 13 columns
            - **Target:** `Mental_Health_Score`
            - **Target range:** 3.6 – 9.4
            - **Key variables:** Stress Level,
              Avg. Daily Usage Hours, Sleep Hours,
              Study Hours, Physical Activity,
              Platform and Purpose
            """
        )


        stat_df = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Avg_Daily_Usage_Hours",
                    "Daily_Unlocks",
                    "Study_Hours",
                    "Physical_Activity_Hours",
                    "Sleep_Hours_Per_Night",
                    "Mental_Health_Score",
                ],

                "Min": [
                    18,
                    1.0,
                    62,
                    0.3,
                    0.0,
                    3.6,
                    3.6,
                ],

                "Mean": [
                    20.82,
                    5.08,
                    171.46,
                    3.01,
                    1.75,
                    6.63,
                    6.23,
                ],

                "Max": [
                    24,
                    8.8,
                    273,
                    8.3,
                    4.1,
                    9.9,
                    9.4,
                ],
            }
        )


        st.dataframe(
            stat_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================

with tab_model:

    st.markdown(
        "<div class='section-header'>"
        "Model Comparison"
        "</div>",
        unsafe_allow_html=True
    )


    with st.container(border=True):

        formatted_metrics = MODEL_METRICS.copy()

        for column in [
            "R2 (Test)",
            "R2 (Train)",
            "MAE",
            "RMSE",
        ]:

            formatted_metrics[column] = (
                formatted_metrics[column]
                .map(lambda x: f"{x:.3f}")
            )


        st.dataframe(
            formatted_metrics,
            use_container_width=True,
            hide_index=True,
        )


    c1, c2 = st.columns(2)


    with c1:

        with st.container(border=True):

            fig = px.bar(
                MODEL_METRICS,
                x="Model",
                y="R2 (Test)",
                color="Model",
                color_discrete_sequence=[
                    "#a1c4fd",
                    "#43e97b",
                    "#38f9d7",
                ],
                title="R² Score (Test Set)",
            )


            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "color": "#e8e8ff"
                },
                showlegend=False,
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with c2:

        with st.container(border=True):

            fig = px.bar(
                MODEL_METRICS,
                x="Model",
                y="MAE",
                color="Model",
                color_discrete_sequence=[
                    "#a1c4fd",
                    "#43e97b",
                    "#38f9d7",
                ],
                title="Mean Absolute Error",
            )


            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={
                    "color": "#e8e8ff"
                },
                showlegend=False,
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with st.container(border=True):

        st.markdown(
            """
            **Deployed model:** Random Forest Regressor

            **Test R²:** 0.877

            **Pipeline steps:**

            1. `Study_Hours` → log transformation →
               standard scaling

            2. Numerical variables →
               standard scaling

            3. `Stress_Level` →
               ordinal encoding

            4. Categorical variables →
               one-hot encoding

            5. Random Forest Regressor →
               final prediction

            The deployed model is stored in
            `Mental_Health_Model.pkl`.
            """
        )


# =========================================================
# TAB 4 — ABOUT PROJECT
# =========================================================

with tab_about:

    with st.container(border=True):

        st.markdown(
            "<div class='section-header'>"
            "Project Overview"
            "</div>",
            unsafe_allow_html=True
        )


        st.markdown(
            """
            **Title:** Student Social Media & Mental Health
            Impact — Predictive Modeling

            **Objective:** Explore how social-media habits
            and lifestyle factors such as sleep, study time,
            physical activity and stress relate to student
            mental well-being.

            The application predicts a
            **Mental Health Score (0–10)** using a trained
            machine-learning regression pipeline.

            **Dataset:** 5,000 student records.

            **Workflow:**

            1. Data cleaning
            2. Exploratory Data Analysis
            3. Feature engineering
            4. Preprocessing pipeline
            5. Model training and comparison
            6. Model serialization with Joblib
            7. Streamlit deployment
            """
        )


    c1, c2 = st.columns(2)


    with c1:

        with st.container(border=True):

            st.markdown(
                "**📁 Features Used**"
            )


            st.markdown(
                """
                - Age
                - Gender
                - Academic Level
                - Country
                - Most Used Platform
                - Purpose of Use
                - Avg. Daily Usage Hours
                - Daily Unlocks
                - Study Hours
                - Physical Activity Hours
                - Sleep Hours
                - Stress Level
                - Mental Health Score
                """
            )


    with c2:

        with st.container(border=True):

            st.markdown(
                "**🛠️ Tech Stack**"
            )


            st.markdown(
                """
                - Python
                - Pandas
                - NumPy
                - Scikit-learn
                - Joblib
                - Streamlit
                - Plotly
                - Matplotlib / Seaborn
                """
            )


    with st.container(border=True):

        st.markdown(
            "**👤 Project Author**"
        )


        col_a, col_b = st.columns(
            [1, 1.4]
        )


        with col_a:

            st.markdown(
                """
                **Name:** Anand Kumar

                **Role:** Data Analyst

                **Education:** B.A. Geography Honours,
                Magadh University, Bihar

                **Certification:** Master of Data Analytics
                Program — The iScale (2026)

                **Location:** Bihar, India

                **Email:** anandpal3244@gmail.com

                **GitHub:** [anandpal3244-coder](https://github.com/anandpal3244-coder)

                **LinkedIn:** [anand-pal](https://www.linkedin.com/in/anand-pal-6a657b393/)
                """
            )


        with col_b:

            st.markdown(
                "**🧩 Other Projects**"
            )


            st.markdown(
                """
                - [Student Career Success Prediction](https://student-career-success-prediction-n4xvvdute4gcwqjrwgyjdp.streamlit.app/)

                - [Movie Recommendation System](https://movie-recommender-jdts3fvmdqq3vril8uh8sd.streamlit.app/)

                - [Retail Sales Performance Dashboard](https://public.tableau.com/app/profile/anand.kumar1736/viz/RetailSalesPerformanceDashboard_17820255179240/RetailSalesPerformanceDashboard?publish=yes)

                - [Music Store SQL Analysis](https://github.com/anandpal3244-coder/Music-Store-SQL-Analysis)
                """
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built with ❤️ by Anand Kumar using Streamlit · "
    "Random Forest regression · Student Social Media & "
    "Mental Health Impact dataset"
)