import streamlit as st
import logging
from composable_app.agents import task_assigner
from composable_app.agents.generic_writer_agent import Writer, WriterFactory
from composable_app.utils.human_feedback import record_human_feedback
from composable_app.pages import patched_asyncio

logger = logging.getLogger(__name__)

@st.cache_resource
def find_writer(topic) -> Writer:
    assigner = task_assigner.TaskAssigner()
    writer = patched_asyncio.run(assigner.find_writer(topic))
    return writer

def assign_to_writer():
    try:
        st.title("Assign Writer")

        topic = st.session_state.topic
        st.write(f"Identifying writer to create content on {topic} ...")
        logger.info(f"Asking task assigner to find agent to write about {topic}")
        options = [writer.name for writer in list(Writer)]
        suggested_writer = find_writer(topic)
        st.write(f"Suggested option is {suggested_writer.name}")
        writer_selection = st.selectbox(label="Choose Writer:",
                                        options=options,
                                        index=list(Writer).index(suggested_writer))

        if st.button("Next"):
            if suggested_writer.name != writer_selection:
                record_human_feedback("assigned_writer", topic, suggested_writer.name, writer_selection)
            writer = Writer(writer_selection)
            st.write(f"Delegating work to {writer.name}")
            st.session_state.writer = WriterFactory.create_writer(writer)
            st.switch_page("pages/2_CreateDraft.py")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    assign_to_writer()
