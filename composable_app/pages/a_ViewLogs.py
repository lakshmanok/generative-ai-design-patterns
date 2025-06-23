import streamlit as st
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)

@st.cache_resource
def get_data(log_name) -> pd.DataFrame:
    source_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(source_dir, f"../{log_name}.log")
    df = pd.read_json(filename, lines=True)
    df = df.iloc[::-1] # reverse order of rows
    return df

def display_logs():
    try:
        st.title("View Logs")
        options = ["guards", "feedback", "prompts"]
        log_selection = st.selectbox(label="Choose Log",
                                     options=options,
                                     index=None)

        if log_selection:
            st.header(log_selection)
            st.dataframe(get_data(log_selection))

        if st.button("Reload"):
            get_data.clear(log_selection)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    display_logs()
