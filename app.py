import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analysis import read_csv_safely, build_summary, build_prompt
from ai_helper import stream_gemini
from local_ai import local_models, stream_local

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analysis Assistant")

uploaded = st.file_uploader("Upload a CSV file", type="csv")
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# ---------- Load ----------
try:
    df = read_csv_safely(uploaded)
except Exception as e:
    st.error(f"Could not read this file as a CSV: {e}")
    st.stop()

if df.empty:
    st.error("The file has no rows.")
    st.stop()

# Charts on huge files are slow, so plot a sample
plot_df = df.sample(20000, random_state=0) if len(df) > 20000 else df
if len(df) > 20000:
    st.caption(f"Large file: charts use a random sample of 20,000 of {len(df):,} rows.")

# ---------- Preview ----------
st.subheader("Preview")
st.dataframe(df.head(10))
c1, c2 = st.columns(2)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])

# ---------- Summary statistics ----------
st.subheader("Summary statistics")
selected = st.multiselect("Columns to show", df.columns.tolist(), default=df.columns.tolist())
if selected:
    chosen = df[selected]
    num_part = chosen.select_dtypes(include="number")
    txt_part = chosen.select_dtypes(exclude="number")
    if not num_part.empty:
        st.caption("Numeric columns")
        st.dataframe(num_part.describe().T)
    if not txt_part.empty:
        st.caption("Text columns")
        st.dataframe(txt_part.describe().T)
else:
    st.info("Select at least one column.")

# ---------- Missing values ----------
st.subheader("Missing values")
missing = df.isnull().sum().rename("missing")
missing = missing[missing > 0]
if missing.empty:
    st.success("No missing values found.")
else:
    st.dataframe(missing)

# ---------- Chart ----------
st.subheader("Chart")
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

chart = st.selectbox(
    "Chart type",
    ["Histogram", "Count", "Bar (average by group)", "Boxplot", "Correlation heatmap"],
)

fig, ax = plt.subplots(figsize=(8, 4))

if chart == "Histogram" and num_cols:
    x = st.selectbox("Numeric column", num_cols)
    sns.histplot(data=plot_df, x=x, kde=True, ax=ax)

elif chart == "Count" and cat_cols:
    x = st.selectbox("Category column", cat_cols)
    n_unique = df[x].nunique()
    order = df[x].value_counts().head(20).index
    sns.countplot(data=plot_df, x=x, order=order, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    if n_unique > 20:
        st.caption(f"Showing the 20 most common of {n_unique:,} values.")

elif chart in ("Bar (average by group)", "Boxplot") and num_cols and cat_cols:
    x = st.selectbox("Group by (category)", cat_cols)
    y = st.selectbox("Value (numeric)", num_cols)
    if df[x].nunique() > 20:
        st.warning(f"'{x}' has {df[x].nunique():,} unique values, too many for a readable chart. Pick another column.")
        st.stop()
    if chart == "Boxplot":
        sns.boxplot(data=plot_df, x=x, y=y, ax=ax)
    else:
        sns.barplot(data=plot_df, x=x, y=y, ax=ax)

elif chart == "Correlation heatmap" and len(num_cols) >= 2:
    sns.heatmap(plot_df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)

else:
    st.warning("This CSV doesn't have the right column types for that chart.")
    st.stop()

st.pyplot(fig)
plt.close(fig)

# ---------- AI summary ----------
st.subheader("AI summary")

col_a, col_b = st.columns([3, 1])
context = col_a.text_input("Optional: what is this dataset about?", placeholder="e.g. Titanic passenger records")
language = col_b.selectbox("Answer language", ["English", "Chinese", "Malay"])

summary = build_summary(df)
with st.expander("Data sent to the AI (statistics only, no raw rows)"):
    st.text(summary)

# ---- model selection ----
installed = local_models()
options = ["Gemini (cloud, auto fallback)"] + [f"Local: {m}" for m in installed]

default = next((i for i, o in enumerate(options) if o.startswith("Local: qwen2.5:7b")), 0)
choice = st.selectbox("AI model", options, index=default)

local_model = choice.removeprefix("Local: ") if choice.startswith("Local: ") else None

if not installed:
    st.caption("Local models appear here when Ollama is running with at least one model installed.")

prompt = build_prompt(summary, context, language)

# ---- generate and show ----
if st.button("Generate AI summary"):
    stream = stream_local(prompt, local_model) if local_model else stream_gemini(prompt)
    with st.spinner("Working... local models can take a while before the first word appears."):
        text = st.write_stream(stream)
    st.session_state["ai_text"] = text
    st.session_state["ai_by"] = local_model or "Gemini"
    st.session_state["ai_file"] = uploaded.name
    st.caption(f"Answered by {st.session_state['ai_by']}")
elif st.session_state.get("ai_file") == uploaded.name:
    st.markdown(st.session_state["ai_text"])
    st.caption(f"Answered by {st.session_state['ai_by']}")