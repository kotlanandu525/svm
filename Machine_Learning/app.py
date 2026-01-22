# app.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Loan Status Prediction - SVM", layout="wide")
st.title("🏦 Loan Status Prediction using SVM")
st.markdown(
    "<h4 style='text-align:center; color:purple;'>Linear vs Polynomial vs RBF Kernel</h4>",
    unsafe_allow_html=True
)

# ---------------------------------
# Load CSS
# ---------------------------------
def load_css(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, file_name)
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found")

load_css("style.css")

# ---------------------------------
# Dataset Loader
# ---------------------------------
st.subheader("📂 Dataset Loader")

uploaded_file = st.file_uploader(
    "Upload Loan Dataset CSV",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("Dataset loaded from uploaded file")
elif os.path.exists("train_u6lujuX_CVtuZ9i.csv"):
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    st.success("Dataset loaded from local file")
else:
    st.warning("Please upload the Loan dataset CSV file to continue.")
    st.stop()

# ---------------------------------
# Dataset Preview
# ---------------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# ---------------------------------
# Preprocessing
# ---------------------------------
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']

label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# ---------------------------------
# Train-Test Split & Scaling
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# Train Models
# ---------------------------------
svm_linear = SVC(kernel='linear', C=1)
svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')

svm_linear.fit(X_train_scaled, y_train)
svm_poly.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)

# Predictions
y_pred_linear = svm_linear.predict(X_test_scaled)
y_pred_poly = svm_poly.predict(X_test_scaled)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

acc_linear = accuracy_score(y_test, y_pred_linear)
acc_poly = accuracy_score(y_test, y_pred_poly)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

# ---------------------------------
# Accuracy Display
# ---------------------------------
st.subheader("📊 Model Accuracy")
st.markdown(
    f"""
    <div style='text-align:center; background-color:#E8F8F5; padding:20px; border-radius:15px;'>
        <h3>Linear Kernel: {acc_linear:.3f}</h3>
        <h3>Polynomial Kernel: {acc_poly:.3f}</h3>
        <h3>RBF Kernel: {acc_rbf:.3f}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Accuracy Plot
# ---------------------------------
st.subheader("📈 Accuracy Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=["Linear", "Polynomial", "RBF"],
    y=[acc_linear, acc_poly, acc_rbf],
    palette="viridis",
    ax=ax
)
ax.set_ylim(0, 1)
for i, v in enumerate([acc_linear, acc_poly, acc_rbf]):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
st.pyplot(fig)

# ---------------------------------
# Confusion Matrix Heatmaps
# ---------------------------------
st.subheader("🔥 Confusion Matrix Heatmaps")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_linear),
            annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Linear Kernel")

sns.heatmap(confusion_matrix(y_test, y_pred_poly),
            annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Polynomial Kernel")

sns.heatmap(confusion_matrix(y_test, y_pred_rbf),
            annot=True, fmt="d", cmap="Oranges", ax=axes[2])
axes[2].set_title("RBF Kernel")

for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

st.pyplot(fig)

# ---------------------------------
# 🔮 Loan Status Prediction (Minimal Inputs)
# ---------------------------------
st.subheader("🔮 Predict Loan Status (Minimal Inputs)")

kernel_choice = st.selectbox(
    "Choose SVM Kernel",
    ["Linear", "Polynomial", "RBF"]
)

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
app_income = st.number_input("Applicant Income", 0, 100000, 5000)
loan_amount = st.number_input("Loan Amount", 0, 1000, 150)
credit_history = st.selectbox("Credit History", [1.0, 0.0])

input_data = {
    "Gender": gender,
    "Married": married,
    "Education": education,
    "ApplicantIncome": app_income,
    "LoanAmount": loan_amount,
    "Credit_History": credit_history
}

input_df = pd.DataFrame([input_data])

# Encode categorical inputs
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# 🔥 Align features
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# 🔥 Scale FULL feature matrix
input_df = pd.DataFrame(
    scaler.transform(input_df),
    columns=X.columns
)

model_map = {
    "Linear": svm_linear,
    "Polynomial": svm_poly,
    "RBF": svm_rbf
}

if st.button("Predict Loan Status"):
    prediction = model_map[kernel_choice].predict(input_df)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    st.markdown(
        f"""
        <div style="text-align:center; background-color:#FDEBD0;
        padding:25px; border-radius:15px;">
            <h2>{result}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
