import chromadb
from typing import List, Dict, Any
from .databaseInterface import DatabaseInterface


class ChromaDB(DatabaseInterface):
    def __init__(
        self,
        collection_name: str = "youtube_transcripts",
        persist_path: str = "chroma_db"
    ):
        # ✅ Persistent DB
        self.client = chromadb.PersistentClient(path=persist_path)

        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "YouTube Transcripts",
                    "hnsw:space": "cosine",
                }
            )

    # -------------------------------
    # ADD EMBEDDING
    # -------------------------------
    def add_embedding(
        self,
        id: str,
        embedding: List[float],
        document: str,
        metadata: Dict[str, Any]
    ):
        try:
            self.collection.add(
                embeddings=[embedding],
                ids=[id],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception as e:
            print("Error adding embedding:", e)

    # -------------------------------
    # FIND SIMILAR
    # -------------------------------
    def find_similar_embeddings(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Dict[str, Any]:

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # 🔍 DEBUG (VERY IMPORTANT)
            print("Chroma Results:", results)

            return results

        except Exception as e:
            print("Error querying DB:", e)
            return {}

    # -------------------------------
    # GET DOCUMENT
    # -------------------------------
    def get_document(self, id: str) -> str:
        result = self.collection.get(ids=[id])
        return result.get('documents', [None])[0]

    # -------------------------------
    # GET METADATA
    # -------------------------------
    def get_metadata(self, id: str) -> Dict[str, Any]:
        result = self.collection.get(ids=[id])
        return result.get('metadatas', [None])[0]

    # -------------------------------
    # 🔥 NEW: COUNT (for debugging)
    # -------------------------------
    def count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0