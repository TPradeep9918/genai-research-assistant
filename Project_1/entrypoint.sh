#!/bin/bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODELS="${OLLAMA_MODELS:-llama3.2 mistral gemma3:1b}"

echo "========================================"
echo "  Research Paper Q&A — Starting up"
echo "========================================"

# ── Wait for Ollama to be ready ───────────────────────────────────────────────
echo "Waiting for Ollama at $OLLAMA_HOST ..."
until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null; do
    sleep 3
done
echo "Ollama is ready."

# ── Pull models if not already present ───────────────────────────────────────
for model in $MODELS; do
    if curl -sf "$OLLAMA_HOST/api/tags" | grep -q "\"$model\""; then
        echo "Model $model already present, skipping pull."
    else
        echo "Pulling model: $model ..."
        curl -sf "$OLLAMA_HOST/api/pull" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$model\"}" \
            | tail -1
        echo "Model $model ready."
    fi
done

# ── Build knowledge base if not already built ─────────────────────────────────
if [ ! -f "bm25_index.pkl" ] || [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo "Building knowledge base from papers/ ..."
    python ingest.py
else
    echo "Knowledge base already exists, skipping ingest."
fi

# ── Start Streamlit ───────────────────────────────────────────────────────────
echo "Starting Streamlit on port 8501 ..."
exec python -m streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.fileWatcherType=none \
    --server.headless=true
