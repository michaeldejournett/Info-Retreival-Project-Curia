FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-download both local models so they're baked into the image.
# Pass --build-arg USE_LOCAL_MODELS_BUILD=false to skip the ~2GB download.
ARG USE_LOCAL_MODELS_BUILD=true
ARG TEMPORAL_MODEL_ID=satyaalmasian/temporal_tagger_roberta2roberta
ARG EXPANSION_MODEL_ID=google/flan-t5-base
ENV HF_HOME=/root/.cache/huggingface
RUN if [ "$USE_LOCAL_MODELS_BUILD" = "true" ]; then \
      TEMPORAL_MODEL_ID="$TEMPORAL_MODEL_ID" EXPANSION_MODEL_ID="$EXPANSION_MODEL_ID" \
      python -c "import os; from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel; \
t_id=os.environ['TEMPORAL_MODEL_ID']; e_id=os.environ['EXPANSION_MODEL_ID']; \
AutoTokenizer.from_pretrained(t_id); EncoderDecoderModel.from_pretrained(t_id); \
AutoTokenizer.from_pretrained(e_id); AutoModelForSeq2SeqLM.from_pretrained(e_id)"; \
    fi

COPY search.py api.py scraper.py local_models.py temporal.py expansion.py timex3.py ./

ENV EVENTS_FILE=scraped/events.json
ENV SCRAPE_INTERVAL=3600
ENV SCRAPE_WORKERS=10
ENV USE_LOCAL_MODELS=false
ENV LOCAL_MODEL_FALLBACK_TO_GEMINI=true

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
