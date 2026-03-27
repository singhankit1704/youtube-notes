from typing import Optional, List
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


class YouTubeUtils:
    def __init__(self):
        # Lightweight embedding model (free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Lightweight text generation model
        self.generator = pipeline(
        "text-generation",
        model="distilgpt2"
      )

    # -------------------------------
    # 1. FETCH TRANSCRIPT (ROBUST)
    # -------------------------------
    def get_transcript(self, video_id: str):
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi().list(video_id)

            # Print available transcripts (for debugging)
            print("Available transcripts:", list(transcript_list))

            # Try English
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                transcript = list(transcript_list)[0]

            data = transcript.fetch()

            return " ".join([item.text for item in data])

        except Exception as e:
            print("FULL ERROR:", str(e))
            return None
    # -------------------------------
    # 2. GENERATE EMBEDDINGS
    # -------------------------------
    def generate_embeddings(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    # -------------------------------
    # 3. GENERATE SUMMARY
    # -------------------------------
    def generate_summary(self, text: str) -> str:
        prompt = f"Summarize:\n{text[:500]}"  # reduce input size

        result = self.generator(
            prompt,
            max_new_tokens=100,   # ✅ IMPORTANT FIX
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]

        return result.replace(prompt, "").strip()

    # -------------------------------
    # 4. QUESTION ANSWERING (RAG STYLE)
    # -------------------------------
    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
    Context: {context[:500]}

    Question: {question}

    Answer:
    """

        result = self.generator(
            prompt,
            max_new_tokens=80,   # ✅ FIX
            do_sample=True,
            temperature=0.5
        )[0]["generated_text"]

        return result.replace(prompt, "").strip()

        # Basic hallucination control
        if len(answer) < 5:
            return "I don't have information about this in the video."

        return answer