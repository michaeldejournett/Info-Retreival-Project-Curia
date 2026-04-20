Adapter expectations:

- Providers: gemini, ollama, huggingface
- Hugging Face backends: `api` (hosted/local endpoint over HTTP) and `local` (in-process transformers)

- Input: raw user query
- Output: JSON-compatible fields
  - keywords: list of strings
  - date_from/date_to: YYYY-MM-DD or null
  - time_from/time_to: HH:MM or null

All adapters normalize and validate outputs before scoring.
