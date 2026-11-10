# J.A.R.V.I.S. — Local Voice + RAG Assistant

FastAPI · ChromaDB · Ollama · MySQL (Laragon) · Web Speech API

## Run
1. Start **MySQL** in Laragon (root / no password, port 3306).
2. Make sure **Ollama** is running and has the models:
   `ollama pull llama3.2` and `ollama pull nomic-embed-text`
3. Double-click `run.bat` (first run creates `venv` and installs `requirements.txt`),
   or run `venv\Scripts\python main.py` and open http://localhost:8000

The `jarvis_db` database and the `chat_sessions` / `messages` tables are created automatically.

## Config
Model names, chunk size, top-k, DB credentials and the system prompt are at the top of `main.py`.

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Index a PDF / TXT / MD file (multipart `file`) |
| POST | `/chat` | `{ "message": "...", "session_id": "..." }` → reply + sources |
| GET | `/history?session_id=` | Messages for a session (latest session if omitted) |
| DELETE | `/history` | Delete all chat history |
| GET | `/documents` · DELETE `/documents/{name}` | List / remove indexed files |
| GET | `/status` | Ollama / MySQL / model health |

## Voice
- **Voice In — Local Whisper (default):** the browser records your mic, detects when you stop
  talking, and sends the clip to `POST /transcribe`, where `faster-whisper` turns it into text on
  your PC. It works offline and in any modern browser. You see a live transcript while you speak,
  and the message is sent automatically when you pause. Change `WHISPER_MODEL` in `main.py`
  (`tiny.en` is faster, `small.en` is more accurate).
- **Voice In — Browser:** the Web Speech API (Chrome/Edge only, needs internet). Switch engines with the
  "Voice In" chip above the input box.
- Click the **mic** or the **orb** to talk. Click again to send early. `Esc` cancels.
- **Voice Output (TTS)** uses `speechSynthesis` and runs fully locally. `Esc` or "Stop Speaking" stops it.
