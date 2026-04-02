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

    # -------------------------------
    # FETCH TRANSCRIPT
    # -------------------------------
    def get_transcript(self, video_id: str):
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)

            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                transcript = list(transcript_list)[0]

            data = transcript.fetch()
            return " ".join([item.text for item in data])

        except Exception as e:
            print("ERROR:", str(e))
            return None

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
{context[:500]}

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