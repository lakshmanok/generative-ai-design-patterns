import streamlit as st
import logging
from composable_app.agents import reviewer_panel
from composable_app.pages import patched_asyncio

logger = logging.getLogger(__name__)

@st.cache_resource
def summarize(topic, draft_title, final_reviews) -> str:
    draft = st.session_state.draft
    assert draft.title == draft_title
    st.write(f"Reviewing {draft_title} on {topic}")
    logger.info(f"Reviewing {draft_title} on {topic}")
    panel_review = patched_asyncio.run(reviewer_panel.summarize_reviews(draft, final_reviews, topic))
    return panel_review

def perform_panel_review():
    try:
        st.title("Panel Review")

        topic = st.session_state.topic
        writer = st.session_state.writer
        draft = st.session_state.draft
        final_reviews = st.session_state.final_reviews

        panel_review = summarize(topic, draft.title, final_reviews)
        st.markdown(panel_review)

        if st.button("Next"):
            st.session_state.panel_review = panel_review
            st.switch_page("pages/6_FinalVersion.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    perform_panel_review()
