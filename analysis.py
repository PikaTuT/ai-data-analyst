import numpy as np
import pandas as pd


def read_csv_safely(file):
    """Read a CSV with automatic delimiter and encoding handling."""
    last_error = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError as e:
            last_error = e
    raise ValueError(f"Could not decode the file: {last_error}")


def build_summary(df, max_cat_cols=5, top_n=5):
    """Turn any DataFrame into a compact text summary for the AI."""
    lines = [f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns."]
    lines.append("Columns: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()))

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("Missing values: none.")
    else:
        lines.append("Missing values (count, % of rows):")
        for c in missing.index:
            lines.append(f"  {c}: {missing[c]} ({missing[c] / len(df) * 100:.1f}%)")

    # Numeric columns
    num = df.select_dtypes(include="number")
    if not num.empty:
        lines.append("Numeric columns:")
        desc = num.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
        lines.append(desc.to_string())

        if num.shape[1] >= 2:
            corr = num.corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            strong = upper[upper.abs() >= 0.5].sort_values(key=abs, ascending=False).head(5)
            if not strong.empty:
                lines.append("Strong correlations (abs >= 0.5):")
                for (a, b), v in strong.items():
                    lines.append(f"  {a} vs {b}: {v:.2f}")

    # Categorical columns
    cat = df.select_dtypes(exclude="number")
    for c in cat.columns[:max_cat_cols]:
        top = df[c].value_counts().head(top_n)
        common = ", ".join(f"{str(k)[:30]} ({v})" for k, v in top.items())
        lines.append(f"{c}: {df[c].nunique()} unique values. Most common: {common}")

    return "\n".join(lines)


def build_prompt(summary, context="", language="English"):
    about = f"What the user says the dataset is about: {context}\n\n" if context.strip() else ""
    return f"""You are a data analyst writing for a student.

{about}Below are summary statistics computed from an uploaded dataset. Write a short analysis.

Rules:
- Use only the numbers provided. Do not invent statistics.
- Give 3 key findings, each with its supporting number.
- Mention one limitation, such as missing values or small groups, and how it could affect the results.
- Do not claim causes; describe patterns only.
- Write the answer in {language}.
- Keep it under 150 words, in plain English.

Statistics:
{summary}
"""