import streamlit as st
import pandas as pd
import pickle
import random
import time
import os

# -------------------------------
# Header & Intro
# -------------------------------
st.header("🎓 Placement Prediction Using Machine Learning")

data = '''
Placement Prediction using Machine Learning helps colleges and students analyze employability.
By inputting academic records and personal details, the model predicts whether a student is likely to get placed.  

**Algorithms Used:**  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Machine  
- XGBoost  
'''

st.markdown(data)

# Project Image (top image)
st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=150)

# -------------------------------
# Load Model & Columns
# -------------------------------
MODEL_FILE = "best_model_Random Forest.pkl"
COLUMNS_FILE = "model_columns.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ {MODEL_FILE} not found! Please ensure it is in the same folder.")
    st.stop()

if not os.path.exists(COLUMNS_FILE):
    st.error("❌ model_columns.pkl not found! Please ensure it is in the same folder.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    placement_model = pickle.load(f)

with open(COLUMNS_FILE, "rb") as f:
    model_columns = pickle.load(f)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📊 Enter Student Details")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135823.png", width=150)

gender = st.sidebar.selectbox("Gender", ["M", "F"])
ssc_p = st.sidebar.slider("SSC Percentage", 30.0, 100.0, 70.0)
hsc_p = st.sidebar.slider("HSC Percentage", 30.0, 100.0, 65.0)
hsc_s = st.sidebar.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
degree_p = st.sidebar.slider("Degree Percentage", 30.0, 100.0, 66.0)
degree_t = st.sidebar.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
workex = st.sidebar.selectbox("Work Experience", ["Yes", "No"])
etest_p = st.sidebar.slider("E-Test Percentage", 0.0, 100.0, 60.0)
specialisation = st.sidebar.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])
mba_p = st.sidebar.slider("MBA Percentage", 30.0, 100.0, 62.0)

# Collect input values
input_data = pd.DataFrame([[gender, ssc_p, hsc_p, hsc_s, degree_p, degree_t,
                            workex, etest_p, specialisation, mba_p]],
                          columns=["gender", "ssc_p", "hsc_p", "hsc_s", "degree_p",
                                   "degree_t", "workex", "etest_p", "specialisation", "mba_p"])

# One-hot encode input
input_data = pd.get_dummies(input_data)

# Align with training columns
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# -------------------------------
# Prediction with Progress Bar
# -------------------------------
st.subheader("🔮 Predicting Placement")

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader("Processing Student Data...")

place = st.empty()
place.image("https://i.makeagif.com/media/1-17-2024/dw-jXM.gif", width=200)

for i in range(100):
    time.sleep(0.03)
    progress_bar.progress(i + 1)

prediction = placement_model.predict(input_data)[0]

placeholder.empty()
place.empty()

# -------------------------------
# Final Output
# -------------------------------
if prediction == 1:
    st.success("🎉 Congratulations! The student is **Placed** ✅")
    st.image("https://images.examples.com/wp-content/uploads/2018/07/student.gif", width=250)
else:
    st.warning("😔 Unfortunately, the student is **Not Placed** ❌")
    st.image("https://i.ytimg.com/vi/88cAJRL5joE/maxresdefault.jpg", width=250)

st.markdown('Designed by: **Niyati Singh and Vanshika Bhardwaj**')
