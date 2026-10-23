"""读取 PDF 手册 → 文本分块 → 写入 ChromaDB 向量库。

用法：python -m src.manual_loader
"""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from . import config

COLLECTION_NAME = "tool_manuals"
EMBEDDING_MODEL = "BAAI/bge-small-zh"


def _split_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_collection():
    """获取（或创建）向量集合；优先中文 Embedding，失败时回退默认模型。"""
    client = chromadb.PersistentClient(path=str(config.INDEX_DIR))
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[manual_loader] {EMBEDDING_MODEL} 不可用，回退默认 Embedding：{exc}")
        ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)


def ingest(manual_dir: Path | None = None) -> int:
    manual_dir = manual_dir or config.MANUAL_DIR
    pdfs = sorted(Path(manual_dir).glob("*.pdf"))
    if not pdfs:
        print(f"[manual_loader] {manual_dir} 下没有 PDF，请先放入检修手册")
        return 0

    collection = get_collection()
    ids, docs, metas = [], [], []
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        for idx, chunk in enumerate(_split_text(text)):
            ids.append(f"{pdf.stem}-{idx}")
            docs.append(chunk)
            metas.append({"source": pdf.name, "chunk": idx})

    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"[manual_loader] 入库完成：{len(ids)} 个片段，来自 {len(pdfs)} 份手册")
    return len(ids)


if __name__ == "__main__":
    ingest()
