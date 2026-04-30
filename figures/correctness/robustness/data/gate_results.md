# Gate Results

Gate profile: `correctness`

| Model | Passed | Failed Checks |
|---|---:|---|
| ollama:llama3:latest | no | date_exact_rate |
| gemini:gemma-3-27b-it | no | parse_success_rate, keyword_f1, date_exact_rate, time_partial_rate |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | yes | - |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | no | keyword_f1 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | yes | - |
| huggingface:google/flan-t5-base | no | keyword_f1, date_exact_rate, time_partial_rate |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | no | keyword_f1, date_exact_rate, time_partial_rate |
