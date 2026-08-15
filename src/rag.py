from pathlib import Path

import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "models" / "chroma_db"

COLLECTION_NAME = "amanleb_cybersecurity"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ==================================================
# TRUSTED SOURCES
# ==================================================

SOURCES = [
    {
        "source_id": "isf_sms_fraud_2026",
        "title": (
            "ISF Warning — Fraudulent SMS "
            "Impersonating a Money Transfer Company"
        ),
        "url": (
            "https://isf.gov.lb/news/"
            "fraudulent-text-messages-impersonating-a-money-transfer-company-"
            "beware-of-the-theft-of-your-personal-data-and-money/"
        ),
        "organization": "Lebanese Internal Security Forces",
    },
    {
        "source_id": "isf_security_awareness",
        "title": "ISF Internet Security Awareness",
        "url": (
            "https://isf.gov.lb/internet-security-awareness/"
        ),
        "organization": "Lebanese Internal Security Forces",
    },
    {
        "source_id": "alfa_security_tips",
        "title": "Alfa Security Tips",
        "url": (
            "https://www.alfa.com.lb/en/support/security-tips"
        ),
        "organization": "Alfa",
    },
]


# ==================================================
# WEB EXTRACTION
# ==================================================

def extract_webpage_text(url: str) -> str:
    """Download a webpage and return normalized visible text."""

    response = requests.get(
        url,
        timeout=30,
        headers=REQUEST_HEADERS,
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
        ]
    ):
        tag.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True,
    )

    return " ".join(text.split())


def extract_between(
    text: str,
    start_marker: str,
    end_marker: str | None = None,
) -> str:
    """Extract a source-specific content region using case-insensitive markers."""

    text_lower = text.lower()

    start_index = text_lower.find(
        start_marker.lower()
    )

    if start_index == -1:
        return text

    cleaned = text[start_index:]

    if end_marker is not None:
        cleaned_lower = cleaned.lower()

        end_index = cleaned_lower.find(
            end_marker.lower()
        )

        if end_index != -1:
            cleaned = cleaned[:end_index]

    return " ".join(cleaned.split())


# ==================================================
# DOCUMENT CLEANING
# ==================================================

def load_cleaned_documents() -> list[dict]:
    """Fetch and clean all trusted-source documents."""

    documents = []

    for source in SOURCES:
        text = extract_webpage_text(
            source["url"]
        )

        if source["source_id"] == "isf_sms_fraud_2026":
            clean_text = extract_between(
                text,
                start_marker=(
                    "Issued by the General Directorate"
                ),
                end_marker="Other Articles",
            )

        elif source["source_id"] == "isf_security_awareness":
            clean_text = extract_between(
                text,
                start_marker=(
                    "Your Behaviour on the Internet"
                ),
            )

        elif source["source_id"] == "alfa_security_tips":
            clean_text = extract_between(
                text,
                start_marker="General tips",
                end_marker="ABOUT ALFA",
            )

        else:
            clean_text = text

        documents.append(
            {
                **source,
                "text": clean_text,
            }
        )

    return documents


# ==================================================
# CHUNKING
# ==================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < chunk_size."
        )

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunks.append(
            " ".join(words[start:end])
        )

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


def create_chunks(
    documents: list[dict],
) -> list[dict]:
    """Create chunks while preserving source metadata."""

    chunks = []

    for doc in documents:
        document_chunks = chunk_text(
            doc["text"]
        )

        for chunk_index, chunk in enumerate(
            document_chunks
        ):
            chunks.append(
                {
                    "chunk_id": (
                        f"{doc['source_id']}_"
                        f"{chunk_index}"
                    ),
                    "source_id": doc["source_id"],
                    "title": doc["title"],
                    "organization": (
                        doc["organization"]
                    ),
                    "url": doc["url"],
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    return chunks


# ==================================================
# RAG ENGINE
# ==================================================

class AmanLebRAG:
    """Multilingual semantic retriever over trusted Lebanese cybersecurity sources."""

    def __init__(self):
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        CHROMA_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

        self.chunks = []

        self._prepare_database()

    def _prepare_database(self) -> None:
        """
        Reuse an existing persistent collection when available.
        Build it from the trusted webpages only when the collection is empty.
        """

        if self.collection.count() > 0:
            self._load_existing_chunks()
            return

        documents = load_cleaned_documents()
        self.chunks = create_chunks(documents)

        embedding_texts = [
            (
                f"Title: {chunk['title']}\n"
                f"Organization: {chunk['organization']}\n"
                f"Content: {chunk['text']}"
            )
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            embedding_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.collection.upsert(
            ids=[
                chunk["chunk_id"]
                for chunk in self.chunks
            ],
            documents=[
                chunk["text"]
                for chunk in self.chunks
            ],
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "source_id": chunk["source_id"],
                    "title": chunk["title"],
                    "organization": (
                        chunk["organization"]
                    ),
                    "url": chunk["url"],
                    "chunk_index": (
                        chunk["chunk_index"]
                    ),
                }
                for chunk in self.chunks
            ],
        )

    def _load_existing_chunks(self) -> None:
        """Reconstruct chunk metadata from an existing persistent Chroma collection."""

        stored = self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        chunks = []

        for chunk_id, document, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
        ):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": metadata["source_id"],
                    "title": metadata["title"],
                    "organization": (
                        metadata["organization"]
                    ),
                    "url": metadata["url"],
                    "chunk_index": (
                        metadata["chunk_index"]
                    ),
                    "text": document,
                }
            )

        self.chunks = sorted(
            chunks,
            key=lambda chunk: (
                chunk["source_id"],
                chunk["chunk_index"],
            ),
        )

    def retrieve(
        self,
        sms: str,
        n_results: int = 1,
    ) -> list[dict]:
        """Retrieve the most semantically relevant trusted-source chunks."""

        collection_size = self.collection.count()

        if collection_size == 0:
            return []

        n_results = min(
            n_results,
            collection_size,
        )

        query_embedding = self.embedding_model.encode(
            [sms],
            normalize_embeddings=True,
        )

        results = self.collection.query(
            query_embeddings=(
                query_embedding.tolist()
            ),
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        for i in range(n_results):
            metadata = (
                results["metadatas"][0][i]
            )

            retrieved.append(
                {
                    "source": metadata["title"],
                    "organization": (
                        metadata["organization"]
                    ),
                    "url": metadata["url"],
                    "source_id": (
                        metadata["source_id"]
                    ),
                    "chunk_index": (
                        metadata["chunk_index"]
                    ),
                    "distance": float(
                        results["distances"][0][i]
                    ),
                    "text": (
                        results["documents"][0][i]
                    ),
                }
            )

        return retrieved


# ==================================================
# SAFE EXTRACTIVE ANALYSIS
# ==================================================

SUSPICIOUS_KEYWORDS = [
    "fraudulent",
    "steal",
    "banking information",
    "personal information",
    "fraud",
]

ACTION_STARTERS = [
    "Do not click",
    "Do not enter",
    "Do not forward",
    "Verify ",
]


def _add_adjacent_context(
    rag: AmanLebRAG,
    top: dict,
) -> str:
    """Append the next chunk from the same source when available."""

    evidence_text = top["text"]

    next_chunk_index = (
        top["chunk_index"] + 1
    )

    for chunk in rag.chunks:
        if (
            chunk["source_id"] == top["source_id"]
            and chunk["chunk_index"] == next_chunk_index
        ):
            evidence_text += " " + chunk["text"]
            break

    return " ".join(
        evidence_text.split()
    )


def _extract_suspicious_sentence(
    evidence_text: str,
) -> str | None:
    """Select the first evidence sentence containing at least two fraud indicators."""

    sentences = (
        evidence_text
        .replace("!", ".")
        .replace("?", ".")
        .split(".")
    )

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_lower = sentence.lower()

        score = sum(
            keyword in sentence_lower
            for keyword in SUSPICIOUS_KEYWORDS
        )

        if score >= 2:
            return sentence + "."

    return None


def _extract_actions(
    evidence_text: str,
) -> list[str]:
    """Extract up to three action statements directly from the retrieved evidence."""

    positions = []

    for starter in ACTION_STARTERS:
        position = evidence_text.find(
            starter
        )

        if position != -1:
            positions.append(
                (position, starter)
            )

    positions.sort()

    actions = []

    for i, (start, _) in enumerate(
        positions
    ):
        if i + 1 < len(positions):
            end = positions[i + 1][0]
        else:
            end = len(evidence_text)

        action = (
            evidence_text[start:end]
            .strip()
            .strip(" ;")
        )

        if action:
            actions.append(action)

    return actions[:3]


def build_safe_analysis(
    rag: AmanLebRAG,
    sms: str,
) -> dict:
    """
    Build a deterministic, source-grounded safety explanation
    from the most relevant retrieved evidence.
    """

    retrieved = rag.retrieve(
        sms,
        n_results=1,
    )

    if not retrieved:
        raise RuntimeError(
            "No trusted-source evidence is available."
        )

    top = retrieved[0]

    evidence_text = _add_adjacent_context(
        rag,
        top,
    )

    return {
        "why_suspicious": (
            _extract_suspicious_sentence(
                evidence_text
            )
        ),
        "actions": (
            _extract_actions(
                evidence_text
            )
        ),
        "source": top["source"],
        "organization": top["organization"],
        "url": top["url"],
        "distance": top["distance"],
    }
