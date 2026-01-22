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
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    file_path = "train_u6lujuX_CVtuZ9i.csv"

    # Case 1: File exists in repo (GitHub / local)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    # Case 2: Upload manually (Streamlit Cloud safe)
    uploaded_file = st.file_uploader(
        "Upload Loan Dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    st.stop()

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
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

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

y_pred_linear = svm_linear.predict(X_test)
y_pred_poly = svm_poly.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

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

# ---------------------------------
# Confusion Matrix Heatmaps
# ---------------------------------
st.subheader("🔥 Confusion Matrix Heatmaps")

cm_linear = confusion_matrix(y_test, y_pred_linear)
cm_poly = confusion_matrix(y_test, y_pred_poly)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[0])
axes[0].set_title("Linear Kernel")

sns.heatmap(cm_poly, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[1])
axes[1].set_title("Polynomial Kernel")

sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[2])
axes[2].set_title("RBF Kernel")

for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

st.pyplot(fig)

