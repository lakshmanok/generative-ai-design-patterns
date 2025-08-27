import streamlit as st
import logging
from agents import reviewer_panel
from pages import patched_asyncio

logger = logging.getLogger(__name__)

@st.cache_resource
def review(topic, draft_title) -> list:
    draft = st.session_state.draft
    assert draft.title == draft_title
    st.write(f"Reviewing {draft_title} on {topic}")
    logger.info(f"Reviewing {draft_title} on {topic}")
    first_round_reviews = patched_asyncio.run(reviewer_panel.do_first_round_reviews(draft, topic))
    return first_round_reviews

def perform_panel_review():
    try:
        st.title("Panel Review (Round 1)")

        topic = st.session_state.topic
        writer = st.session_state.writer
        draft = st.session_state.draft

        first_round_reviews = review(topic, draft.title)
        st.write(first_round_reviews)

        if st.button("Next"):
            st.session_state.first_round_reviews = first_round_reviews
            st.switch_page("pages/4_PanelReview2.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    perform_panel_review()
