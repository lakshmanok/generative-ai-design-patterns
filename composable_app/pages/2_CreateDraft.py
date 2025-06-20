import streamlit as st
import logging
import asyncio
from composable_app.agents.article import Article

logger = logging.getLogger(__name__)

@st.cache_resource
def write_about(writer_name, topic) -> Article:
    writer = st.session_state.writer
    assert writer.name() == writer_name # this is so that writer_name is part of the caching
    st.write(f"Employing {writer.name()} to create content on {topic} ...")
    logger.info(f"Employing {writer.name()} to create content on {topic} ...")

    article = asyncio.run(writer.write_about(topic))
    return article

def write_draft():
    try:
        st.title("Create draft")

        topic = st.session_state.topic
        writer = st.session_state.writer
        draft = write_about(writer.name(), topic)
        draft_textbox = st.markdown(draft.to_markdown())

        if st.button("Next"):
            st.session_state.draft = draft
            st.switch_page("pages/3_PanelReview.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    write_draft()
