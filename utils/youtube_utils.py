from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class YouTubeUtils:
    def __init__(self):
        # ✅ Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # ✅ LOAD FLAN-T5 MANUALLY (NO PIPELINE)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        # ✅ YouTube Transcript API instance
        self.ytt_api = YouTubeTranscriptApi()

    @staticmethod
    def extract_video_id(url_or_id: str) -> str:
        import re
        url_or_id = url_or_id.strip()
        if len(url_or_id) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', url_or_id):
            return url_or_id
        match = re.search(r'(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/|^)([A-Za-z0-9_-]{11})', url_or_id)
        if match:
            return match.group(1)
        return url_or_id

    # -------------------------------
    # FETCH TRANSCRIPT
    # -------------------------------
    def get_transcript(self, video_id: str):
        video_id = self.extract_video_id(video_id)
        try:
            # ✅ NEW API: Use fetch() directly with language preference
            fetched_transcript = self.ytt_api.fetch(video_id, languages=['en'])
            return " ".join([snippet.text for snippet in fetched_transcript])

        except Exception as e:
            print(f"English transcript not found, trying any available language...")
            try:
                # Fallback: fetch without language filter (defaults to English,
                # but may return auto-generated)
                fetched_transcript = self.ytt_api.fetch(video_id)
                return " ".join([snippet.text for snippet in fetched_transcript])
            except Exception as e2:
                print("ERROR fetching transcript:", str(e2))
                raise ValueError(f"Failed to fetch transcript for video {video_id}: {str(e2)}")

    # -------------------------------
    # EMBEDDINGS
    # -------------------------------
    def generate_embeddings(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    # -------------------------------
    # 🔥 SUMMARY (ENGLISH ONLY)
    # -------------------------------
    def generate_summary(self, text: str) -> str:
        try:
            prompt = f"""
Summarize the following content in simple English only:

{text[:500]}
"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120
            )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            print("ERROR in summary:", e)
            return "Error generating summary."

    # -------------------------------
    # 🔥 ANSWER (RAG)
    # -------------------------------
    def generate_answer(self, question: str, context: str) -> str:
        try:
            prompt = f"""
Answer the question using ONLY the context below.
Respond in English only.

Context:
{context}

Question:
{question}
"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120
            )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if len(answer.strip()) < 5:
                return "I couldn't find a clear answer in the video."

            return answer.strip()

        except Exception as e:
            print("ERROR in answer:", e)
            return "Error generating answer."