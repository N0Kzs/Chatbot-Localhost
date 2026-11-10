"""
J.A.R.V.I.S. — local Voice + RAG assistant.
FastAPI + ChromaDB + Ollama + MySQL (Laragon).

Run:  venv\\Scripts\\python main.py      then open http://localhost:8000
"""

# ============================================================================
# CONFIGURATION — edit these to taste
# ============================================================================
OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "jarvis_db"

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "jarvis_docs"

TOP_K = 3                 # chunks retrieved per question
CHUNK_SIZE = 1000         # characters per chunk
CHUNK_OVERLAP = 200       # characters shared between neighbouring chunks
HISTORY_TURNS = 6         # previous messages sent to the model for context
MAX_DISTANCE = 0.75       # cosine distance cut-off; chunks further away are ignored

# Local speech-to-text (faster-whisper). Sizes: tiny.en, base.en, small.en, medium.en
# Bigger = more accurate but slower. Model downloads once (~150 MB for base.en).
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cpu"          # "cuda" if you have an NVIDIA GPU + CUDA libs
WHISPER_COMPUTE = "int8"        # "float16" on GPU

HOST = "127.0.0.1"
PORT = 8000

SYSTEM_PROMPT = """You are J.A.R.V.I.S., a calm, precise and slightly witty AI assistant.
Answer concisely (usually 1-4 sentences unless detail is requested).
Your replies may be read aloud, so prefer plain sentences over tables or long lists.
When the CONTEXT below is relevant, base your answer on it and mention the source file name.
If the context does not contain the answer, say so briefly and answer from general knowledge.

CONTEXT:
{context}"""

# ============================================================================
import io
import json
import tempfile
import threading
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from sqlalchemy import (DateTime, ForeignKey, Integer, String, Text,
                        create_engine, text)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("jarvis")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ALLOWED_EXT = {".pdf", ".txt", ".md"}

# ----------------------------------------------------------------------------
# MySQL
# ----------------------------------------------------------------------------
SERVER_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}"
DB_URL = f"{SERVER_URL}/{MYSQL_DB}?charset=utf8mb4"


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))           # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT, "mysql"))
    source_documents: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    session: Mapped[ChatSession] = relationship(back_populates="messages")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "sources": json.loads(self.source_documents) if self.source_documents else [],
            "timestamp": self.timestamp.isoformat(),
        }


engine = None
SessionLocal = None


def init_db() -> bool:
    """Create jarvis_db + tables if missing. Returns True when MySQL is usable."""
    global engine, SessionLocal
    if SessionLocal is not None:
        return True
    try:
        server = create_engine(SERVER_URL, pool_pre_ping=True)
        with server.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        server.dispose()
        engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        log.info("MySQL ready: database '%s'", MYSQL_DB)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("MySQL unavailable (%s). Is Laragon's MySQL started?", e.__class__.__name__)
        return False


def db() -> Session:
    if not init_db():
        raise HTTPException(503, "MySQL is not reachable on "
                                 f"{MYSQL_HOST}:{MYSQL_PORT}. Start MySQL in Laragon.")
    return SessionLocal()


# ----------------------------------------------------------------------------
# Ollama + ChromaDB
# ----------------------------------------------------------------------------
llm = ollama.Client(host=OLLAMA_HOST)
chroma = chromadb.PersistentClient(path=str(BASE_DIR / CHROMA_PATH))
collection = chroma.get_or_create_collection(
    CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def embed(texts: list[str]) -> list[list[float]]:
    try:
        out = []
        for i in range(0, len(texts), 32):          # batch to keep requests small
            out.extend(llm.embed(model=EMBED_MODEL, input=texts[i:i + 32]).embeddings)
        return out
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Embedding failed via Ollama ({EMBED_MODEL}): {e}")


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(txt: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph/sentence boundaries where possible, with overlap."""
    txt = "\n".join(line.rstrip() for line in txt.replace("\r\n", "\n").split("\n"))
    txt = txt.strip()
    chunks, start, n = [], 0, len(txt)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = txt[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = txt[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    if collection.count() == 0:
        return []
    res = collection.query(query_embeddings=embed([query]), n_results=min(k, collection.count()),
                           include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= MAX_DISTANCE:
            hits.append({"source": meta.get("source"), "chunk": meta.get("chunk"),
                         "score": round(1 - dist, 3), "text": doc})
    return hits


# ----------------------------------------------------------------------------
# Whisper (local speech-to-text) — loaded lazily on first use
# ----------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_whisper = None
_whisper_lock = threading.Lock()


def whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            log.info("Loading Whisper model '%s' on %s…", WHISPER_MODEL, WHISPER_DEVICE)
            _whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        return _whisper


def load_audio(path: str, sr: int = 16000):
    """Decode any audio file (webm/ogg/wav/mp3/mp4) to 16 kHz mono float32.
    Tolerates truncated files, which the live preview sends while you're still talking."""
    import av
    import numpy as np
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    parts = []
    try:
        with av.open(path) as container:
            try:
                for frame in container.decode(audio=0):
                    parts += [f.to_ndarray() for f in resampler.resample(frame)]
            except av.error.FFmpegError:
                pass                                    # partial clip: keep what decoded
            parts += [f.to_ndarray() for f in resampler.resample(None)]
    except av.error.FFmpegError:
        pass
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=1).flatten().astype(np.float32) / 32768.0


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    threading.Thread(target=whisper, daemon=True).start()   # warm up STT in background
    log.info("Chroma collection '%s' holds %d chunks", CHROMA_COLLECTION, collection.count())
    yield


app = FastAPI(title="J.A.R.V.I.S.", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/status")
def status():
    try:
        names = {m.model.split(":")[0] for m in llm.list().models}
        ollama_ok = True
    except Exception:  # noqa: BLE001
        names, ollama_ok = set(), False
    return {
        "ollama": ollama_ok,
        "chat_model": CHAT_MODEL, "chat_model_ready": CHAT_MODEL.split(":")[0] in names,
        "embed_model": EMBED_MODEL, "embed_model_ready": EMBED_MODEL.split(":")[0] in names,
        "mysql": init_db(),
        "chunks": collection.count(),
        "whisper": _whisper is not None, "whisper_model": WHISPER_MODEL,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "Only PDF, TXT and MD files are supported.")
    data = await file.read()
    try:
        txt = extract_text(name, data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read {name}: {e}")
    chunks = chunk_text(txt)
    if not chunks:
        raise HTTPException(400, f"No extractable text found in {name} (scanned PDF?).")

    vectors = embed(chunks)
    collection.delete(where={"source": name})          # re-upload replaces old version
    collection.add(
        ids=[f"{name}::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": name, "chunk": i, "uploaded": datetime.now().isoformat()}
                   for i in range(len(chunks))],
    )
    log.info("Indexed %s → %d chunks", name, len(chunks))
    return {"filename": name, "chunks": len(chunks), "characters": len(txt)}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Local speech-to-text: accepts a webm/ogg/wav/mp3 clip, returns the text."""
    data = await audio.read()
    if len(data) < 1000:
        return {"text": ""}
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        def run():
            samples = load_audio(path)
            if samples.size < 16000 * 0.3:          # under 0.3 s: nothing to transcribe
                return ""
            segments, _ = whisper().transcribe(
                samples, language="en", beam_size=1, vad_filter=True,
                condition_on_previous_text=False)
            return " ".join(seg.text.strip() for seg in segments).strip()
        from starlette.concurrency import run_in_threadpool
        text = await run_in_threadpool(run)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return {"text": text}


@app.get("/documents")
def documents():
    data = collection.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for m in data["metadatas"]:
        d = docs.setdefault(m["source"], {"filename": m["source"], "chunks": 0,
                                          "uploaded": m.get("uploaded")})
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d.get("uploaded") or "", reverse=True)


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    collection.delete(where={"source": filename})
    return {"deleted": filename}


@app.post("/chat")
def chat(body: ChatIn):
    question = body.message.strip()
    if not question:
        raise HTTPException(400, "Empty message.")

    s = db()
    try:
        session = s.get(ChatSession, body.session_id) if body.session_id else None
        if session is None:
            session = ChatSession(id=body.session_id or str(uuid.uuid4()), title=question[:80])
            s.add(session)
            s.flush()
        prior = (s.query(Message).filter_by(session_id=session.id)
                 .order_by(Message.id.desc()).limit(HISTORY_TURNS).all())[::-1]

        hits = retrieve(question)
        context = "\n\n---\n\n".join(f"[Source: {h['source']} #{h['chunk']}]\n{h['text']}"
                                     for h in hits) or "(no relevant documents found)"
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        messages += [{"role": m.role, "content": m.content} for m in prior]
        messages.append({"role": "user", "content": question})

        try:
            reply = llm.chat(model=CHAT_MODEL, messages=messages).message.content.strip()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Ollama chat failed ({CHAT_MODEL}): {e}")

        sources = [{"source": h["source"], "chunk": h["chunk"], "score": h["score"],
                    "snippet": h["text"][:300]} for h in hits]
        s.add(Message(session_id=session.id, role="user", content=question))
        s.add(Message(session_id=session.id, role="assistant", content=reply,
                      source_documents=json.dumps(sources) if sources else None))
        s.commit()
        return {"reply": reply, "sources": sources, "session_id": session.id}
    finally:
        s.close()


@app.get("/history")
def history(session_id: Optional[str] = None, limit: int = 200):
    """Messages of one session; without session_id, the most recent session."""
    s = db()
    try:
        if not session_id:
            latest = s.query(ChatSession).order_by(ChatSession.created_at.desc()).first()
            session_id = latest.id if latest else None
        if not session_id:
            return {"session_id": None, "messages": []}
        msgs = (s.query(Message).filter_by(session_id=session_id)
                .order_by(Message.id.asc()).limit(limit).all())
        return {"session_id": session_id, "messages": [m.to_dict() for m in msgs]}
    finally:
        s.close()


@app.delete("/history")
def clear_history():
    """Delete all chat sessions and messages."""
    s = db()
    try:
        n = s.query(Message).delete()
        s.query(ChatSession).delete()
        s.commit()
        return {"deleted_messages": n}
    finally:
        s.close()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    os.chdir(BASE_DIR)
    uvicorn.run(app, host=HOST, port=PORT)
