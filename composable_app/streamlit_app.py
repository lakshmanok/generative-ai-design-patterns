import logging

## make sure logging starts first
def setup_logging(config_file: str = "logging.json"):
    import json
    import logging.config

    # Load the JSON configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Apply the configuration
    logging.config.dictConfig(config)
setup_logging()
##

import streamlit as st

def app_main():
    # Set the page configuration for a wider layout
    st.set_page_config(
        page_title="Educational Workbook Creation",
        page_icon="🏠",
        layout="wide")

    # Define the pages
    page_0 = st.Page("pages/0_SelectTopic.py", title="Select topic", icon="🏠")
    page_1 = st.Page("pages/1_AssignToWriter.py", title="Assign to Writer", icon="✒️")
    page_2 = st.Page("pages/2_CreateDraft.py", title="Create Draft", icon="✍️")
    page_3 = st.Page("pages/3_PanelReview.py", title="Panel Review", icon="🤼")
    page_4 = st.Page("pages/4_FinalVersion.py", title="Final Version", icon="✒️")

    # Set up navigation
    pg = st.navigation([page_0, page_1, page_2, page_3, page_4])

    # Run the selected page
    pg.run()

if __name__ == "__main__":
    app_main()