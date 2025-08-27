import streamlit as st
import logging

from agents import reviewer_panel
from pages import patched_asyncio

logger = logging.getLogger(__name__)

@st.cache_resource
def review(topic, draft_title, first_round_reviews) -> list:
    draft = st.session_state.draft
    assert draft.title == draft_title

    st.write(f"Reviewing {draft_title} on {topic} for second time")
    logger.info(f"Reviewing {draft_title} on {topic} for second time")
    first_round_reviews = patched_asyncio.run(reviewer_panel.do_second_round_reviews(draft, first_round_reviews, topic))
    return first_round_reviews

def perform_panel_review():
    try:
        st.title("Panel Review (Round 2)")

        topic = st.session_state.topic
        writer = st.session_state.writer
        draft = st.session_state.draft
        first_round_reviews = st.session_state.first_round_reviews

        final_reviews = review(topic, draft.title, first_round_reviews)
        st.write(final_reviews)

        if st.button("Next"):
            st.session_state.final_reviews = final_reviews
            st.switch_page("pages/5_SummarizeReview.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    perform_panel_review()
