import streamlit as st
import logging
import asyncio
from composable_app.agents import reviewer_panel

logger = logging.getLogger(__name__)

@st.cache_resource
def review(topic, draft_title) -> str:
    draft = st.session_state.draft
    assert draft.title == draft_title
    st.write(f"Reviewing {draft_title} on {topic}")
    logger.info(f"Reviewing {draft_title} on {topic}")
    panel_review = asyncio.run(reviewer_panel.get_panel_review_of_article(topic, draft))
    return panel_review

def perform_panel_review():
    try:
        st.title("Panel Review")

        topic = st.session_state.topic
        writer = st.session_state.writer
        draft = st.session_state.draft

        panel_review = review(topic, draft.title)
        st.markdown(panel_review)

        if st.button("Next"):
            st.session_state.panel_review = panel_review

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    perform_panel_review()
