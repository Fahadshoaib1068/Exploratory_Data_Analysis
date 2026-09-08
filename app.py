import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exploratory Data Analysis Interface", layout="wide")

# ---------------------------------------------------------------
# SIDEBAR: Dataset Controls (file upload + attribute selection)
# ---------------------------------------------------------------
st.sidebar.header("Dataset Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis", type=["csv"])

st.title("Exploratory Data Analysis Interface")

if uploaded_file is not None:
    # Validate that the file is a properly formatted CSV
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read the uploaded file as a valid CSV. Error: {e}")
        st.stop()

    # -----------------------------------------------------------
    # MAIN CONTENT - TOP SECTION: Dataset Preview & Metadata
    # -----------------------------------------------------------
    st.subheader("Dataset Preview & Metadata")

    st.markdown("**First 5 Rows:**")
    st.dataframe(df.head())

    st.markdown(f"**Shape:** {df.shape}  (rows: {df.shape[0]}, columns: {df.shape[1]})")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Column Data Types:**")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    with col2:
        st.markdown("**Missing Values per Column:**")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
        st.dataframe(missing_df)

    st.markdown("**Basic Statistical Summary (numerical attributes):**")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary = df[numeric_cols].agg(["mean", "median", "min", "max"]).T
        st.dataframe(summary)
    else:
        st.info("No numerical columns found in this dataset.")

    st.divider()

    # -----------------------------------------------------------
    # SIDEBAR: Attribute Selection
    # -----------------------------------------------------------
    st.sidebar.header("Attribute Selection")
    selected_col = st.sidebar.selectbox("Select Attribute for Visualization", df.columns)

    # -----------------------------------------------------------
    # MAIN CONTENT - BOTTOM SECTION: Conditional Visualization
    # -----------------------------------------------------------
    st.subheader("Visualization")

    if selected_col:
        is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])

        if is_numeric:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df[selected_col].dropna(), bins=20, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogram of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            counts = df[selected_col].value_counts()
            pct = (counts / counts.sum() * 100).round(1)

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(counts.index.astype(str), counts.values, color="lightgreen", edgecolor="black")
            ax.set_title(f"Bar Chart of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency Count")
            plt.xticks(rotation=45, ha="right")

            # optional percentage display on top of bars
            for bar, p in zip(bars, pct.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{p}%", ha="center", va="bottom", fontsize=8)

            st.pyplot(fig)

else:
    st.info("Upload a CSV file from the sidebar to begin exploring the dataset. "
            "(Tested with titanic.csv)")