from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from app.config import get_settings


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

@dataclass
class PolicyChunk:
    content: str
    product: str
    section: str
    score: float


def _load_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))
    return pages


def _split_text_into_chunks(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    chunks: List[str] = []
    n = len(text)
    start = 0
    step = max_chars - overlap

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step

    return chunks


def _get_chroma_client():
    settings = get_settings()
    persist_dir = settings.vector_db_dir
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client



def _get_policy_collection():
    
    client = _get_chroma_client()

    collection = client.get_or_create_collection(
        name="travel_insurance_policies",
    )
    return collection


def build_policy_index(force_rebuild: bool = False):
    settings = get_settings()
    collection = _get_policy_collection()

    if force_rebuild:
        collection.delete(where={})

    pdfs = [
        ("EUROPAX", DATA_DIR / "notice_europax.pdf"),
        ("GLOBE TRAVELLER", DATA_DIR / "notice_globe.pdf"),
    ]

    print("Building policy index...")

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for product_name, pdf_path in pdfs:
        pages = _load_pdf_text(pdf_path)
        for page_num, page_text in pages:
            chunks = _split_text_into_chunks(page_text)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{product_name.lower().replace(' ', '_')}_p{page_num}_c{idx}"
                ids.append(doc_id)
                texts.append(chunk)
                metadatas.append(
                    {
                        "product": product_name,
                        "section": f"page:{page_num}",
                    }
                )

    if not texts:
        raise RuntimeError("No policy text found to index.")

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {len(texts)} chunks into 'travel_insurance_policies'.")

def retrieve_policy_chunks(
    question: str,
    top_k: int = 5,
) -> List[PolicyChunk]:
    
    collection = _get_policy_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: List[PolicyChunk] = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        product = meta.get("product", "UNKNOWN")
        section = meta.get("section", "")
        score = float(1.0 / (1.0 + dist)) if dist is not None else 0.0
        chunks.append(
            PolicyChunk(
                content=doc,
                product=product,
                section=section,
                score=score,
            )
        )

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks
