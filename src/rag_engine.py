"""检修手册检索。"""
from .manual_loader import get_collection


class ManualRetriever:
    def __init__(self):
        self.collection = get_collection()

    def query(self, query: str, top_k: int = 3) -> list[dict]:
        result = self.collection.query(query_texts=[query], n_results=top_k)
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        return [
            {
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "score": round(1 - dist, 4) if dist is not None else None,
            }
            for doc, meta, dist in zip(documents[0], metadatas[0], distances[0])
        ]
