FROM python:3.11-slim

# Run as a non-root user on Render
RUN useradd -m -u 1000 user

# Memory controls for a ~512 MB instance: fewer malloc arenas, single-threaded ONNX/BLAS
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

RUN mkdir -p /home/user/app && chown -R user:user /home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .


EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]