FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch keeps the image small; sentence-transformers works with it
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the first visitor does not wait
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV USAGE_DB=/data/usage.sqlite3
VOLUME /data
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
