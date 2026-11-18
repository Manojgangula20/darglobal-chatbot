FROM python:3.11-slim

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Bake the embedding model + index for the committed snapshot (never fails the build)
RUN python -m app.cli build || echo "index pre-build skipped"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
