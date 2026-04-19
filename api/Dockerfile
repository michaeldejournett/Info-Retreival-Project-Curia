FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-download the local instruction model so it's baked into the image.
# Pass --build-arg USE_LOCAL_MODELS_BUILD=false to skip the ~2GB download.
ARG USE_LOCAL_MODELS_BUILD=true
ARG LOCAL_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
ENV HF_HOME=/root/.cache/huggingface
RUN if [ "$USE_LOCAL_MODELS_BUILD" = "true" ]; then \
      LOCAL_MODEL_ID="$LOCAL_MODEL_ID" \
      python -c "import os; from transformers import AutoTokenizer, AutoModelForCausalLM; \
m=os.environ['LOCAL_MODEL_ID']; \
AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"; \
    fi

COPY search.py api.py scraper.py local_models.py temporal.py expansion.py prompt_templates.py ./

ENV EVENTS_FILE=scraped/events.json
ENV SCRAPE_INTERVAL=3600
ENV SCRAPE_WORKERS=10
ENV USE_LOCAL_MODELS=false
ENV LOCAL_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
ENV LOCAL_MODEL_FALLBACK_TO_GEMINI=true

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
