import streamlit as st
import logging
import asyncio
from composable_app.agents.article import Article

logger = logging.getLogger(__name__)

@st.cache_resource
def write_about(writer_name, topic, draft_title, panel_review) -> Article:
    writer = st.session_state.writer
    draft = st.session_state.draft
    assert writer.name() == writer_name # this is so that writer_name is part of the caching
    assert draft.title == draft_title

    st.write(f"Using {writer.name()} to rewrite {draft_title} on {topic} based on panel review")
    logger.info(f"Using {writer.name()} to rewrite {draft_title} on {topic} based on panel review")

    article = asyncio.run(writer.revise_article(topic, draft, panel_review))
    return article

def final_version():
    try:
        st.title("Final version")

        topic = st.session_state.topic
        writer = st.session_state.writer
        draft = st.session_state.draft
        panel_review = st.session_state.panel_review
        final_article = write_about(writer.name(), topic, draft.title, panel_review)
        st.markdown(final_article.to_markdown())

        if st.button("Next"):
            st.session_state.final_article = final_article
            st.switch_page("pages/4_FinalVersion.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    final_version()
