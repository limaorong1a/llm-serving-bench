# Baseline Report (week 1)

## Environment
- GPU: (5090 )
- Driver/CUDA: (from nvidia-smi)
- vLLM version: (pip show vllm)
- Model: Qwen/Qwen2.5-7B-Instruct (served as: qwen2.5-7b)

## Workload
- Endpoint: /v1/chat/completions (OpenAI-compatible)
- Prompt: fixed short prompt
- max_tokens: 128
- temperature/top_p: 0.7 / 0.8

## Results
### Concurrency = 1
{
  "mode": "nonstream",
  "base_url": "http://127.0.0.1:8000",
  "endpoint": "/v1/chat/completions",
  "model": "qwen2.5-7b",
  "concurrency": 1,
  "requests": 20,
  "wall_time_s": 25.657072584028356,
  "requests_per_second": 0.7795121573008329,
  "error_count": 0,
  "error_rate": 0.0,
  "http_status_counts": {
    "200": 20
  },
  "latency": {
    "mean_s": 1.2826514878950548,
    "p50_s": 1.2909178359550424,
    "p95_s": 1.2959453509713057,
    "p99_s": 1.3129582541913258,
    "min_s": 1.0923463010694832,
    "max_s": 1.317211479996331
  },
  "tokens": {
    "completion_tokens_total": 2540,
    "tokens_per_second": 98.99804397720578,
    "completion_tokens_available_requests": 20
  }
}

### Concurrency = 4
{
  "mode": "nonstream",
  "base_url": "http://127.0.0.1:8000",
  "endpoint": "/v1/chat/completions",
  "model": "qwen2.5-7b",
  "concurrency": 4,
  "requests": 40,
  "wall_time_s": 13.381715213065036,
  "requests_per_second": 2.9891534353493494,
  "error_count": 0,
  "error_rate": 0.0,
  "http_status_counts": {
    "200": 40
  },
  "latency": {
    "mean_s": 1.3338258049741853,
    "p50_s": 1.3308601145399734,
    "p95_s": 1.3635407443682197,
    "p99_s": 1.364281822828343,
    "min_s": 1.2857498310040683,
    "max_s": 1.364404486026615
  },
  "tokens": {
    "completion_tokens_total": 5113,
    "tokens_per_second": 382.0885378735306,
    "completion_tokens_available_requests": 40
  }
}

### Concurrency = 16
{
  "mode": "nonstream",
  "base_url": "http://127.0.0.1:8000",
  "endpoint": "/v1/chat/completions",
  "model": "qwen2.5-7b",
  "concurrency": 16,
  "requests": 80,
  "wall_time_s": 7.462014142074622,
  "requests_per_second": 10.720966012235142,
  "error_count": 0,
  "error_rate": 0.0,
  "http_status_counts": {
    "200": 80
  },
  "latency": {
    "mean_s": 1.468841454538051,
    "p50_s": 1.4702357084606774,
    "p95_s": 1.5246867409674452,
    "p99_s": 1.5261298695160077,
    "min_s": 1.2698631790699437,
    "max_s": 1.5271498439833522
  },
  "tokens": {
    "completion_tokens_total": 10206,
    "tokens_per_second": 1367.7272390108983,
    "completion_tokens_available_requests": 80
  }
}

