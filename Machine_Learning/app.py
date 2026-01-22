# app.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df = load_data()

# ---------------------------------
# Dataset Preview
# ---------------------------------
if st.checkbox("Show Dataset"):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ---------------------------------
# Column Identification
# ---------------------------------
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# ---------------------------------
# Handle Missing Values
# ---------------------------------
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# ---------------------------------
# Feature / Target Split
# ---------------------------------
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# ---------------------------------
# Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------
# Scaling
# ---------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------
# Train Models
# ---------------------------------
svm_linear = SVC(kernel='linear', C=1)
svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')

svm_linear.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

acc_linear = accuracy_score(y_test, svm_linear.predict(X_test))
acc_poly = accuracy_score(y_test, svm_poly.predict(X_test))
acc_rbf = accuracy_score(y_test, svm_rbf.predict(X_test))

# ---------------------------------
# Accuracy Display
# ---------------------------------
st.subheader("📊 Model Accuracy")

st.markdown(
    f"""
    <div style='text-align:center; background-color:#E8F8F5; padding:20px; border-radius:15px;'>
        <h3>Linear Kernel Accuracy: {acc_linear:.3f}</h3>
        <h3>Polynomial Kernel Accuracy: {acc_poly:.3f}</h3>
        <h3>RBF Kernel Accuracy: {acc_rbf:.3f}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Accuracy Comparison Plot
# ---------------------------------
st.subheader("📈 Accuracy Comparison")

accuracies = {
    "Linear Kernel": acc_linear,
    "Polynomial Kernel": acc_poly,
    "RBF Kernel": acc_rbf
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=list(accuracies.keys()),
    y=list(accuracies.values()),
    palette="viridis",
    ax=ax
)

ax.set_ylim(0, 1)
ax.set_xlabel("Kernel Type")
ax.set_ylabel("Accuracy")
ax.set_title("SVM Kernel Accuracy Comparison")

for i, v in enumerate(accuracies.values()):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

st.pyplot(fig)
