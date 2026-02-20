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

# --- Page Configuration ---
st.set_page_config(
    page_title="Social Factors and Health Outcomes",
    page_icon="🎓",
    layout="wide",
)

# --- GLOBAL STYLE: BIGGER FONT ---
st.markdown("""
<style>
/* Make all main text bigger */
html, body, [class*="css"] {
    font-size: 22px;
}

/* Make headings bigger */
h1 {
    font-size: 40px;
}
h2 {
    font-size: 32px;
}
h3 {
    font-size: 28px;
}
h4 {
    font-size: 24px;
}

/* Make table text bigger */
thead tr th {
    font-size: 22px !important;
}
tbody tr td {
    font-size: 20px !important;
}

/* Increase padding inside the page */
div.block-container {
    padding: 3rem 5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Topic", "Objective", "EDA", "Regression"])


# ---------------------- Tab 1: Project Topic ----------------------
with tab1:
    # Background
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926");
        background-size: cover;
        background-attachment: fixed;
    }
    div.block-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 2rem;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # College name and project info
    st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>Modern College of Arts, Commerce and Science</h1>
    <h2 style='text-align: center; color: #2E86C1;'>Shivajinagar, Pune-05</h2>
    <h3 style='text-align: center; color: #117A65;'>Department of Statistics</h3>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <h2 style='text-align: center; color: #AF7AC5;'>Project Title</h2>
    <h1 style='text-align: center; color: #884EA0;'> "A Comprehensive Study of Social Factors and Health Outcomes in India" </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style='text-align: center;'>Presented by</h3>
    <h4 style='text-align: center;'>Trushna Shelke-2434331</h4>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style='text-align: center;'>Under the Guidance of</h3>
    <h4 style='text-align: center;'>Mrs. Gauri Kulkarni</h4>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 18px;'>Academic Year 2024-2025</p>", unsafe_allow_html=True)


# ---------------------- Tab 2: Objective and Dataset ----------------------
with tab2:
    # Background
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926");
        background-size: cover;
        background-attachment: fixed;
    }
    div.block-container {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 2rem;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Load data
    data = pd.read_excel("C:\\New folder (4)\\final_data.xlsx")

    st.title("📈 Objectives And Introduction to Data")

    # Objectives
    st.header("🎯 Project Objectives")
    st.markdown("""
    - **Clustering of Indian States based on Empowerment Indicators**  
    - **Impact of Lifestyle Habits on Men's Health in India**  
    - **Influence of Women's Life Circumstances on Child Health Outcomes**
    """)

    # About data
    st.header("📊 About the Data")

    st.subheader("🔹 Short Background")
    st.markdown("""
    This project examines women's empowerment across rural and urban India, focusing on women aged 15–49 using NFHS-5 data.  
    It covers literacy, marriage age, decision-making, economic independence, and digital inclusion.  
    By comparing rural and urban trends, the study highlights how women's empowerment impacts broader health outcomes.
    """)

    st.subheader("🔹 Source of Data")
    st.markdown("""
    - **National Family Health Survey (NFHS-5)**  
    - **NFHS website**
    """)

    # Preview data
    st.header("🔎 Dataset Preview")
    st.dataframe(data.head())

    # Dataset dimensions
    st.header("📏 Dataset Dimensions")
    st.success(f"The dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**.")

    # Summary Statistics
    st.header("📈 Summary Statistics")
    st.dataframe(data.describe())

    # Download button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        data.to_excel(writer, index=False)

    st.download_button(
        label="💾 Download Dataset as Excel",
        data=buffer.getvalue(),
        file_name='final_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.markdown("---")
    st.caption("Developed as part of MSc Statistics Project | © 2025")



# ---------------------- Tab 3: EDA - Power BI ----------------------
import streamlit as st

with tab3:
    st.header("📊 Power BI Dashboard Overview")
    st.markdown("Here is a glimpse of the Power BI dashboard used in our project:")

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([2, 1])  # The numbers indicate the relative width of the columns

    with col1:
        # Display the Power BI dashboard image
        st.image("C:\\Users\\Admin\\Pictures\\Screenshots\\Screenshot (93).png", caption="Power BI Dashboard Overview", width=1200)

        with open("C:\\New folder (4)\\Dashboard.pbix", "rb") as file:
            st.download_button(
                label="📥 Download and Open Power BI Dashboard",
                data=file,
                file_name="Dashboard.pbix",
                mime="application/octet-stream"
            )

    with col2:
        # Display the conclusion in the second column
        st.subheader("📊 Dashboard Conclusion")

        # Apply styling for the conclusion
        st.markdown("""
        <style>
        ul {
            font-size: 18px;
        }
        li {
            margin-bottom: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Conclusion content
        st.markdown("""
        - **1️⃣ Currently Married Women Participating in Household Decisions:**  
          Participation in three major household decisions has steadily increased across states, highlighting a positive trend in women’s domestic empowerment.

        - **2️⃣ Women Owning House or Land (States/UTs):**  
          Significant variations exist across states, with some regions showing strong female property ownership while others still lag behind.

        - **3️⃣ Women Experiencing Physical Violence During Pregnancy:**  
          Around **61%** of women experience violence during pregnancy, emphasizing a critical social challenge that needs urgent action.

        - **4️⃣ Women Aged 20-24 Married Before Age 18:**  
          Child marriage remains a concern in states like **West Bengal** and **Bihar**, although many areas are showing encouraging declines.

        - **5️⃣ Women Owning House/Land (Specific States):**  
          States such as **Telangana** and **Ladakh** exhibit higher percentages of women owning property, supporting greater financial independence.

        - **6️⃣ Women Having Mobile Phones and Savings Accounts:**  
          Access to mobile phones and banking services is notably high in **Karnataka** and **Telangana**, driving digital and financial empowerment.
        """)


# ---------------------- Tab 4: Regression ----------------------
with tab4:
    st.title("📈 Influence of Women's Life Circumstances on Child Health Outcomes")

    uploaded_file = "C:\\New folder (4)\\final_data.xlsx"

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
                    st.subheader(f"Scatter Plot: {y_var} vs {x_var}")
                    fig, ax = plt.subplots()
                    ax.scatter(X, y, color='blue', label='Data Points')
                    ax.plot(X, y_pred, color='red', label='Regression Line')
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                    ax.set_title(f"{y_var} vs {x_var}")
                    ax.legend()
                    st.pyplot(fig)

                with col2:
                    st.subheader("Regression Interpretation")
                    st.markdown(f"""
                    - *Interpretation: For every 1 unit increase in **{x_var}**, **{y_var}** {direction} by about *{abs(beta1):.2f}* units.
                    - *R² = {r2:.2f}* (explains {r2*100:.1f}% variation in {y_var})
                    - *P-value = {pval:.4f}* → {("Statistically significant ✅" if pval < 0.05 else "Not statistically significant ❌")}
                    """)

                    with st.expander(f"Show Full Regression Summary for {y_var}"):
                        st.text(model.summary())

                st.subheader(f"Predict {y_var} Based on New {x_var} Value")
                new_x = st.number_input(f"Enter a new value for {x_var}:", value=float(X.mean()), key=f"predict_{y_var}")
                new_y = model.predict([1, new_x])[0]
                st.success(f"Predicted {y_var} = {new_y:.2f}")
    else:
        st.info("Please upload your Excel file to proceed.")

