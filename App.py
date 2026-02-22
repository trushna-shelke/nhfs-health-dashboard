# --- Basic Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import statsmodels.api as sm
from scipy.stats import bartlett, ttest_ind, chi2_contingency, kstest, lognorm, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm 
from statsmodels.formula.api import ols
import streamlit as st
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Social Factors and Health Outcomes",
    page_icon="🎓",
    layout="wide",
)

# ---------- Load background ----------
def get_base64(file):
    if os.path.exists(file):
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# Ensure these lines are BEFORE the st.markdown block
bg_path = "background.png" 
bg = get_base64(bg_path)

logo_path = "TS_logo.png"
logo = get_base64(logo_path)
st.markdown(f"""
<style>
.logo-overlay {{
    position: fixed;
    top: 20px;
    left: 2px;
    opacity: 0.95;
    z-index: 999;
}}
</style>

<div class="logo-overlay">
    <img src="data:image/png;base64,{logo}" width="250">
</div>
""", unsafe_allow_html=True)

# Correct way to write the markdown block
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
/* Footer */
.footer {{
    position: fixed;
    bottom: 40px;
    right: 25px;
    font-size: 17px;
    font-weight: 700;
    color: #000000;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Author - Trushna Shelke | '
    '<a href="https://www.linkedin.com/in/trushna-shelke-6b9a69230" target="_blank">LinkedIn</a>'
    '</div>',
    unsafe_allow_html=True
)

# Create space equal to header height
st.markdown(
    "<div style='height:120px'></div>",
    unsafe_allow_html=True
)
st.markdown(f"""
<style>

/* 1. Pin the Tabs (Topic, Objective, etc.) */
div[data-baseweb="tab-list"] {{
    position: fixed !important;
    top: 70px !important;      /* Keeps them at the current height */
    margin-left: 260px !important;
    z-index: 1000 !important;
    background-color: transparent !important;
    width: auto !important;
    display: flex !important;
    
}}
/* Fixed tabs */
div[data-baseweb="tab-list"] {{
    position: fixed !important;
    top: 70px !important;
    margin-left: 260px !important;
    z-index: 9999 !important;
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(6px);
}}

/* 2. Pin the Underline Border Line */
div[data-baseweb="tab-border"] {{
    position: fixed !important;
    top: 115px !important;       /* Pins it exactly under the tab text */
    margin-left: 100px !important;
    z-index: 999 !important;
    width: 70% !important;      /* Matches the look in your screenshot */
    height: 1px !important;
    background-color: rgba(0,0,0,0.2) !important;
}}

/* 3. This pushes your main content down so it doesn't hide behind the fixed tabs */
.block-container {{
    padding-top: 40px !important;
}}
.block-container {{
    padding-top: 160px !important;
}}
</style>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Topic", "Objective", "EDA", "Regression"])


# ---------------------- Tab 1: Project Topic ----------------------
with tab1:
    # Correct way to write the markdown block
    st.markdown(f"""
    <style>
    /* This centers the "Data Analytics Portfolio" line */
    .brand {{
        font-size: 40px; 
        font-weight: 800;
        color: #0f172a;
        text-align: center; 
        width: 100%;
        margin-top: -10px;
    }}

    /* This pushes the box down so it doesn't hide your logo */
    .glass {{
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.8);
        padding: 15px 25px; 
        border-radius: 20px;
        margin-top: 7px; 
        text-align: center;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }}

    /* This keeps the title on ONE line */
    .title {{
        font-size: 26px; 
        font-weight: 850;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap; 
    }}

    /* Tightens spacing to ensure it fits on one page */
    .content-section {{
        margin-top: 15px;
        padding-left: 10%;
    }}

    .module-list {{
        margin-top: 10px;
        list-style-type: none;
        padding-left: 0;
    }}

    .module-list li {{
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
    }}

    .module-list li::before {{
        content: "•";
        color: #2563eb;
        font-weight: bold;
        display: inline-block; 
        width: 1em;
        margin-left: 0;

    </style>
    """, unsafe_allow_html=True)
    # Brand Name (Aligned Right)
    st.markdown('<div class="brand">Data Analytics Portfolio</div>', unsafe_allow_html=True)

    # Glass Box (Centered & Narrower)
    st.markdown("""
    <div class="glass">
        <div class="title">Social Factors & Health Outcomes</div>
    </div>
    """, unsafe_allow_html=True)

    # Description & Vertical Bullets
    with st.container():
        st.markdown("""
    <div class="content-section">
    <div class="subtitle"><b>Case Study Overview</b></div>

    <div class="desc-text" style="margin-bottom: 5px;">
    Explore how social determinants influence public health outcomes through interactive analytics powered by NFHS data.
    </div>

    <ul class="module-list" style="margin-top: 5px;">
    <b>Core Modules</b></li>
    <li>Data Exploration</li>
    <li>Visualization</li>
    <li>Statistical Modeling</li>
    <li>Insight Storytelling</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- Tab 2: Objective and Dataset ----------------------
with tab2:


    # -------- Load data --------
    data = pd.read_excel("final_data.xlsx")

    # -------- Title --------
    st.markdown("<h1 style='font-size:26px;'>📈 Objectives And Introduction to Data</h1>", unsafe_allow_html=True)

    # -------- Objectives --------
    st.markdown("<h2 style='font-size:22px;'>🎯 Project Objectives</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Clustering of Indian States based on Empowerment Indicators  
    - Impact of Lifestyle Habits on Men's Health in India  
    - Influence of Women's Life Circumstances on Child Health Outcomes
    """)

    # -------- About data --------
    st.markdown("<h2 style='font-size:22px;'>📊 About the Data</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size:18px;'>🔹 Short Background</h3>", unsafe_allow_html=True)
    st.markdown("""
    This project examines women's empowerment across rural and urban India, focusing on women aged 15–49 using NFHS-5 data.  
    It covers literacy, marriage age, decision-making, economic independence, and digital inclusion.  
    By comparing rural and urban trends, the study highlights how women's empowerment impacts broader health outcomes.
    """)

    st.markdown("<h3 style='font-size:18px;'>🔹 Source of Data</h3>", unsafe_allow_html=True)
    st.markdown("""
    - National Family Health Survey (NFHS-5)  
    - NFHS website
    """)

    # -------- Preview --------
    st.markdown("<h2 style='font-size:22px;'>🔎 Dataset Preview</h2>", unsafe_allow_html=True)
    st.dataframe(data.head())

    # -------- Dimensions --------
    st.markdown("<h2 style='font-size:22px;'>📏 Dataset Dimensions</h2>", unsafe_allow_html=True)
    st.success(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

    # -------- Summary --------
    st.markdown("<h2 style='font-size:22px;'>📈 Summary Statistics</h2>", unsafe_allow_html=True)
    st.dataframe(data.describe())

    # -------- Download --------
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        data.to_excel(writer, index=False)

    st.download_button(
        label="💾 Download Dataset as Excel",
        data=buffer.getvalue(),
        file_name='final_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# ---------------------- Tab 3: EDA - Power BI ----------------------
import streamlit as st

with tab3:
    
    st.markdown("<h1 style='font-size:26px; font-weight:700;'>📊 Power BI Dashboard Overview</h1>", unsafe_allow_html=True)

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([2.2, 1.1])  # The numbers indicate the relative width of the columns
    
    with col1:
        # Display the Power BI dashboard image
        st.image("Screenshot.png", caption="Power BI Dashboard Overview", width=700)

    with col2:
     st.markdown("""
    <div style="
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.92);
        padding: 25px;
        border-radius: 40px;
        margin-top: 7px;
        box-shadow: 0 15px 25px rgba(0,0,0,0.15);
        line-height: 1.7;
    ">
    <h3 style="margin-top:0;">📊 Dashboard Conclusion</h3>

    <p>1️⃣ Currently Married Women Participation improving</p>
    <p>2️⃣ Women Property Ownership varies across states</p>
    <p>3️⃣ Violence during pregnancy remains a concern</p>
    <p>4️⃣ Child marriage declining but present</p>
    <p>5️⃣ Property ownership linked to empowerment</p>
    <p>6️⃣ Digital inclusion increasing nationwide</p>

    </div>
    """, unsafe_allow_html=True)
    
# ---------------------- Tab 4: Regression ----------------------
with tab4:
    st.markdown('<div class="reg-tab">', unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:26px; font-weight:700;'>📈 Influence of Women's Life Circumstances on Child Health Outcomes</h1>", unsafe_allow_html=True)

    uploaded_file = "final_data.xlsx"

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()

        # Variables
        independent_vars = [
            "Households using clean fuel for cooking3 (%)",
            "Women (age 15-49) who are literate4 (%)",
            "Women (age 15-49)  with 10 or more years of schooling (%)",
            "Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)"
        ]

        dependent_vars = [
            "Neonatal mortality rate (per 1000 live births)",
            "Infant mortality rate (per 1000 live births)",
            "Under-five mortality rate (per 1000 live births)",
            "Children under 5 years who are stunted (height-for-age)18 (%)",
            "Children under 5 years who are wasted (weight-for-height)18 (%)",
            "Children under 5 years who are severely wasted (weight-for-height)19 (%)",
            "Children under 5 years who are underweight (weight-for-age)18 (%)",
            "Children under 5 years who are overweight (weight-for-height)20 (%)"
        ]

        # Dropdowns
        x_var = st.selectbox("Select Independent Variable (X-axis):", independent_vars)
        y_vars = st.multiselect("Select Dependent Variable(s) (Y-axis):", dependent_vars)

        if x_var and y_vars:
            data = df[[x_var] + y_vars].dropna()
            X = data[x_var]
            X_const = sm.add_constant(X)

            for y_var in y_vars:
                y = data[y_var]

                model = sm.OLS(y, X_const).fit()
                y_pred = model.predict(X_const)

                beta1 = model.params[1]
                r2 = model.rsquared
                pval = model.pvalues[1]
                direction = "decreases" if beta1 < 0 else "increases"

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"<h3 style='font-size:18px;'>Scatter Plot: {y_var} vs {x_var}</h3>", unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.scatter(X, y, color='blue', label='Data Points')
                    ax.plot(X, y_pred, color='red', label='Regression Line')
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                    ax.set_title(f"{y_var} vs {x_var}")
                    ax.legend()
                    st.pyplot(fig)

                with col2:
                    st.markdown("<h3 style='font-size:18px;'>Regression Interpretation</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                    - *Interpretation: For every 1 unit increase in **{x_var}**, **{y_var}** {direction} by about *{abs(beta1):.2f}* units.
                    - *R² = {r2:.2f}* (explains {r2*100:.1f}% variation in {y_var})
                    - *P-value = {pval:.4f}* → {("Statistically significant ✅" if pval < 0.05 else "Not statistically significant ❌")}
                    """)

                    with st.expander(f"Show Full Regression Summary for {y_var}"):
                        st.text(model.summary())

                st.subheader(f"Predict {y_var} Based on New {x_var} Value")
                st.markdown(f"<h3 style='font-size:18px;'>Predict {y_var} Based on New {x_var} Value", unsafe_allow_html=True)
                new_x = st.number_input(f"Enter a new value for {x_var}:", value=float(X.mean()), key=f"predict_{y_var}")
                new_y = model.predict([1, new_x])[0]
                st.success(f"Predicted {y_var} = {new_y:.2f}")
    else:
        st.info("Please upload your Excel file to proceed.")
 
    st.markdown('</div>', unsafe_allow_html=True)