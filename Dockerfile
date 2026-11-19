FROM python:3.11-slim

# Hugging Face Spaces / Render: run as a non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1

# WORKDIR alone would create this folder as root, so create it as user's first
RUN mkdir -p /home/user/app && chown -R user:user /home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Bake the embedding model + index for the committed snapshot
RUN python -m app.cli build || echo "index pre-build skipped"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]