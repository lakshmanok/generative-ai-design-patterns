import streamlit as st

def act_on_topic_selection(topic: str):
    st.write(f"Will start to create content on {topic}")
    st.session_state.topic = topic
    st.switch_page("pages/1_AssignToWriter.py")

def select_topic():
    try:
        st.title("Select Topic")
        topic_input = st.text_input(label="Topic")

        st.write("""
        Examples:
        - Battle of the Bulge
        - Compare Kemal Ataturk and Mohammed Jinnah in terms of cultural influence
        - Solve for x:  x^2 - x = 12
        - Squaring the circle
        - Is Pluto a planet?
        - What is in-context learning?
        """)
        if st.button("Next"):
            act_on_topic_selection(topic_input.strip())

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    select_topic()