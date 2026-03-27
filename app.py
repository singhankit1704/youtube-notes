import streamlit as st
from youtube_notes import YouTubeNotes
from database.chroma import ChromaDB


def main():
    st.set_page_config(
        page_title="YouTube AI Notes",
        page_icon="🎥",
        layout="wide"
    )

    # -------------------------------
    # HEADER
    # -------------------------------
    st.title("🎥 YouTube AI Notes (RAG System)")
    st.markdown("""
    Extract insights from YouTube videos using AI.

    ✅ Fetch transcripts  
    ✅ Generate summaries  
    ✅ Ask questions from video content  
    """)

    # -------------------------------
    # INIT SYSTEM
    # -------------------------------
    database = ChromaDB()
    notes_system = YouTubeNotes(database)

    # -------------------------------
    # SIDEBAR: VIDEO PROCESSING
    # -------------------------------
    with st.sidebar:
        st.header("📺 Process Video")

        video_id = st.text_input(
            "Enter YouTube Video ID",
            placeholder="e.g. dQw4w9WgXcQ"
        )

        if st.button("🚀 Process Video"):
            if not video_id:
                st.warning("Please enter a video ID")
            else:
                with st.spinner("Fetching transcript & processing..."):
                    try:
                        result = notes_system.add_video(video_id)

                        if not result:
                            st.error("❌ No transcript available for this video.")
                        else:
                            st.success("✅ Video processed successfully!")

                            st.subheader("📄 Summary")
                            st.write(result["summary"])

                    except Exception:
                        st.error("❌ This video may not have subtitles or is restricted.")
                        print("Transcript:", transcript[:200] if transcript else "None")

    # -------------------------------
    # MAIN: QUESTION ANSWERING
    # -------------------------------
    st.header("💬 Ask Questions")

    question = st.text_input(
        "Ask something about the video",
        placeholder="e.g. What is the main topic of this video?"
    )

    if st.button("🔍 Get Answer"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching for answer..."):
                try:
                    result = notes_system.ask_question(question)

                    st.subheader("🧠 Answer")
                    st.write(result["answer"])

                    if result["context"]:
                        with st.expander("📚 View Context Used"):
                            st.write(result["context"][:3])

                except Exception:
                    st.error("❌ Please process a video first.")

    # -------------------------------
    # FOOTER
    # -------------------------------
    st.markdown("---")
    st.markdown("⚡ Built with Streamlit + RAG + HuggingFace")


if __name__ == "__main__":
    main()