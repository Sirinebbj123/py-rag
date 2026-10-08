
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from metadata_extractor import build_document, extract_invoice

COLLECTION_NAME = "invoices"
DEFAULT_DB_PATH = "qdrant_db"
DEFAULT_DATA_DIR = "1000+ PDF_Invoice_Folder"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # local, sans clé API
VECTOR_SIZE = 384
BATCH_SIZE = 64

_embedder: TextEmbedding | None = None


def embed(texts: list[str]) -> list[list[float]]:
    """Transforme des textes en vecteurs (modèle téléchargé au premier appel)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return [vector.tolist() for vector in _embedder.embed(texts)]


def get_client(db_path: str = DEFAULT_DB_PATH) -> QdrantClient:
    """Serveur Qdrant si QDRANT_URL est défini, sinon base locale sur disque (sans Docker)."""
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=db_path)


def create_collection(client: QdrantClient, reset: bool = False) -> None:
    """Crée la collection "invoices" si elle n'existe pas."""
    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )


def ingest(data_dir: str, db_path: str, reset: bool) -> None:
    pdfs = sorted(Path(data_dir).rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Aucun PDF trouvé dans {data_dir!r}")

    client = get_client(db_path)
    create_collection(client, reset=reset)
    batch: list[tuple[int, str, dict]] = []
    skipped_empty, failed, ingested = [], [], 0

    def flush() -> None:
        nonlocal ingested
        if not batch:
            return
        vectors = embed([document for _, document, _ in batch])
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector=vector, payload={**meta, "document": document})
                    for (point_id, document, meta), vector in zip(batch, vectors)],
        )
        ingested += len(batch)
        batch.clear()
        print(f"  {ingested} factures ingérées...")

    for pdf in pdfs:
        try:
            extracted = extract_invoice(pdf)
        except Exception as error:  # un PDF illisible ne doit pas arrêter le lot
            failed.append((pdf.name, str(error)))
            continue
        meta = extracted["metadata"]
        if meta.pop("is_empty"):
            skipped_empty.append(meta["source"])  # gabarit vide (total 0, aucun article)
            continue
        batch.append((int(meta["invoice_number"]), build_document(extracted), meta))
        if len(batch) >= BATCH_SIZE:
            flush()
    flush()

    total = client.count(COLLECTION_NAME, exact=True).count
    where = os.environ.get("QDRANT_URL") or f"local : {db_path}"
    print(f"\nCollection {COLLECTION_NAME!r} : {total} points ({where})")
    print(f"PDF lus : {len(pdfs)} | ingérés : {ingested} | "
          f"vides ignorés : {len(skipped_empty)} | en erreur : {len(failed)}")
    for name, error in failed:
        print(f"  ERREUR {name} : {error}")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="dossier contenant les PDF")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="dossier de la base Qdrant locale")
    parser.add_argument("--reset", action="store_true", help="supprime la collection avant d'ingérer")
    args = parser.parse_args()
    ingest(args.data_dir, args.db_path, args.reset)