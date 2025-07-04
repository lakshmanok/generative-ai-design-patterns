import streamlit as st
import logging
import dataclasses
from composable_app.agents.article import Article
from composable_app.utils.human_feedback import record_human_feedback
import composable_app.utils.long_term_memory as ltm
from composable_app.pages import patched_asyncio

logger = logging.getLogger(__name__)

def keywords_to_string(a: Article) -> str:
    return '\n'.join(a.index_keywords)

@st.cache_resource
def write_about(writer_name, topic) -> Article:
    writer = st.session_state.writer
    assert writer.name() == writer_name # this is so that writer_name is part of the caching
    st.write(f"Employing {writer.name()} to create content on {topic} ...")
    logger.info(f"Employing {writer.name()} to create content on {topic} ...")

    article = patched_asyncio.run(writer.write_about(topic))
    return article

def write_draft():
    try:
        st.title("Create draft")

        topic = st.session_state.topic
        writer = st.session_state.writer
        ai_generated_draft = write_about(writer.name(), topic)

        if "ai_generated_draft" not in st.session_state or st.session_state.ai_generated_draft != ai_generated_draft:
            # new topic perhaps; this draft was generated for the first time, and not pulled from cache
            st.session_state.ai_generated_draft = ai_generated_draft
            st.session_state.draft = dataclasses.replace(ai_generated_draft) # makes a copy

        # display current version
        draft_title = st.text_area(label="Title", value=st.session_state.draft.title)
        draft_lesson = st.text_area(label="Lesson", value=st.session_state.draft.key_lesson)
        draft_text = st.text_area(label="Text", value=st.session_state.draft.full_text, height=400)
        draft_keywords = st.text_area(label="Keywords", value=keywords_to_string(st.session_state.draft), height=150)

        # allow user to modify through a chat interface also
        st.subheader("Modify draft")
        def modify_draft():
            modify_instruction = st.session_state.modify_instruction
            logger.info(f"Updating draft to instructions: {modify_instruction}")
            draft = patched_asyncio.run(writer.revise_article(topic, st.session_state.draft, modify_instruction))
            # add this instruction to the long-term memory, to use in the future for this user
            ltm.add_to_memory(modify_instruction, metadata={
                "topic": topic,
                "writer": writer.name()
            })
            logger.info(draft.full_text)
            st.session_state.draft = draft  # but keep the original as "ai_generated_draft"
            # because this is a callback, it redraws the page

        with st.form("Modification form", clear_on_submit=True):
            st.text_input(label="Modification instructions", value="", key="modify_instruction")
            st.form_submit_button(label="Modify", on_click=modify_draft)


        if st.button("Next"):
            # grab from UI
            st.session_state.draft.title = draft_title
            st.session_state.draft.key_lesson = draft_lesson
            st.session_state.draft.full_text = draft_text
            st.session_state.draft.index_keywords = draft_keywords.split('\n')
            # has it changed?
            if st.session_state.draft != st.session_state.ai_generated_draft:
                record_human_feedback("initial_draft",
                                      ai_input=topic,
                                      ai_response=st.session_state.ai_generated_draft,
                                      human_choice=st.session_state.draft)
                logger.info(f"User has changed the draft to {st.session_state.draft}")
            st.switch_page("pages/3_PanelReview1.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    write_draft()
