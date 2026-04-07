from typing import Dict, Any, List
from database.databaseInterface import DatabaseInterface
from utils.youtube_utils import YouTubeUtils


class YouTubeNotes:
    def __init__(self, database: DatabaseInterface):
        self.database = database
        self.utils = YouTubeUtils()

    # -------------------------------
    # 🔥 FIX 1: CHUNKING (VERY IMPORTANT)
    # -------------------------------
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    # -------------------------------
    # ADD VIDEO
    # -------------------------------
    def add_video(self, video_id: str) -> Dict[str, Any]:
        transcript = self.utils.get_transcript(video_id)

        if not transcript:
            raise ValueError(f"Failed to fetch transcript for video {video_id}")

        # ✅ Chunk transcript
        chunks = self.chunk_text(transcript)

        # ✅ Generate summary once
        summary = self.utils.generate_summary(transcript)

        # ✅ Store multiple embeddings
        for i, chunk in enumerate(chunks):
            embedding = self.utils.generate_embeddings(chunk)

            metadata = {
                "video_id": video_id,
                "chunk_id": i
            }

            self.database.add_embedding(
                id=f"{video_id}_{i}",
                embedding=embedding,
                document=chunk,
                metadata=metadata
            )

        return {
            "video_id": video_id,
            "summary": summary,
            "chunks": len(chunks),
            "status": "success"
        }

    # -------------------------------
    # ADD MANUAL TRANSCRIPT (FALLBACK)
    # -------------------------------
    def add_manual_transcript(self, transcript: str, source_name: str = "manual") -> Dict[str, Any]:
        """Process a manually pasted transcript (fallback when YouTube blocks IP)."""
        if not transcript or len(transcript.strip()) < 10:
            raise ValueError("Transcript is too short or empty.")

        # ✅ Chunk transcript
        chunks = self.chunk_text(transcript)

        # ✅ Generate summary once
        summary = self.utils.generate_summary(transcript)

        # ✅ Store multiple embeddings
        for i, chunk in enumerate(chunks):
            embedding = self.utils.generate_embeddings(chunk)

            metadata = {
                "video_id": source_name,
                "chunk_id": i
            }

            self.database.add_embedding(
                id=f"{source_name}_{i}",
                embedding=embedding,
                document=chunk,
                metadata=metadata
            )

        return {
            "video_id": source_name,
            "summary": summary,
            "chunks": len(chunks),
            "status": "success"
        }

    # -------------------------------
    # ASK QUESTION
    # -------------------------------
    def ask_question(self, question: str) -> Dict[str, Any]:
        try:
            query_embedding = self.utils.generate_embeddings(question)

            results = self.database.find_similar_embeddings(query_embedding)

            # 🔍 DEBUG
            print("Results:", results)

            if not results or not results.get("documents"):
                return {
                    "answer": "No relevant information found.",
                    "context": []
                }

            # ✅ FIX: Handle nested list safely
            documents = results["documents"][0]

            if not documents:
                return {
                    "answer": "No relevant context found.",
                    "context": []
                }

            # ✅ Use top 3 chunks
            top_docs = documents[:3]

            context = "\n".join(top_docs)

            answer = self.utils.generate_answer(question, context)

            return {
                "answer": answer,
                "context": top_docs,
                "video_ids": results.get("ids", [[]])[0]
            }

        except Exception as e:
            print("ERROR in ask_question:", e)
            return {
                "answer": "Error generating answer.",
                "context": []
            }