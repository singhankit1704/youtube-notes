import streamlit as st
from youtube_notes import YouTubeNotes
from database.chroma import ChromaDB


# -------------------------------
# ✅ CACHED INITIALIZATION
# -------------------------------
@st.cache_resource
def load_database():
    return ChromaDB()


@st.cache_resource
def load_notes_system(_database):
    return YouTubeNotes(_database)


@st.cache_data
def process_video(_notes_system, video_id):
    return _notes_system.add_video(video_id)


# -------------------------------
# MAIN APP
# -------------------------------
def main():
    st.set_page_config(
        page_title="YouTube AI Notes",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 YouTube AI Notes (RAG Chat System)")

    # -------------------------------
    # INIT SYSTEM
    # -------------------------------
    database = load_database()
    notes_system = load_notes_system(database)

    # -------------------------------
    # SESSION STATE
    # -------------------------------
    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -------------------------------
    # SIDEBAR
    # -------------------------------
    with st.sidebar:
        st.header("📺 Process Video")

        video_id = st.text_input("Enter YouTube Video ID")

        if st.button("🚀 Process Video"):
            if not video_id:
                st.warning("Please enter a video ID")
            else:
                with st.spinner("Processing video..."):
                    try:
                        result = process_video(notes_system, video_id)

                        if not result:
                            st.error("❌ No transcript available.")
                        else:
                            st.success("✅ Video processed!")
                            st.session_state.video_processed = True

                            st.subheader("📄 Summary")
                            st.write(result["summary"])

                    except Exception as e:
                        st.error("❌ Error processing video")
                        print(e)

        # Clear chat button
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []

    # -------------------------------
    # CHAT UI
    # -------------------------------
    st.header("💬 Chat with Video")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        if not st.session_state.video_processed:
            st.warning("⚠️ Please process a video first.")
            return

        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = notes_system.ask_question(user_input)
                    answer = result["answer"]

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error("❌ Error generating answer")
                    print(e)

    # -------------------------------
    # FOOTER
    # -------------------------------
    st.markdown("---")
    st.markdown("⚡ Chat-based RAG System with Streamlit")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()