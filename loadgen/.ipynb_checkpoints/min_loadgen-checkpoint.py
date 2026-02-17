import asyncio
import time
import statistics
import json
import argparse
from collections import Counter
from typing import Optional, Tuple, Dict, Any, List
from loadgen.metrics import percentile, summarize_latencies, tokens_per_second
import httpx

PROMPT = "你是一个严谨的助手。请用要点回答：什么是连续批处理（continuous batching）？"

async def one_request_nonstream(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[float, Optional[int], int, Optional[str]]:
    """
    Returns:
      latency_s, completion_tokens, http_status, error_str
    """
    t0 = time.perf_counter()
    try:
        r = await client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            },
            timeout=600,
        )
        t1 = time.perf_counter()
        status = r.status_code
        r.raise_for_status()

        data = r.json()
        usage = data.get("usage", {}) or {}
        out_tokens = usage.get("completion_tokens", None)
        latency = t1 - t0
        return latency, out_tokens if isinstance(out_tokens, int) else None, status, None
    except Exception as e:
        t1 = time.perf_counter()
        # If response exists, try to extract status
        status = getattr(getattr(e, "response", None), "status_code", 0) or 0
        return (t1 - t0), None, status, repr(e)


async def one_request_stream(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[float, Optional[float], Optional[float], Optional[int], int, Optional[str]]:
    """
    Streaming mode: measure TTFT and TPOT.
    Returns:
      latency_s, ttft_s, tpot_s, completion_tokens, http_status, error_str

    Notes:
    - We parse SSE 'data:' lines.
    - completion_tokens may be absent in stream responses depending on server;
      we try to read final usage if present.
    """
    t0 = time.perf_counter()
    ttft = None
    first_token_time = None
    end_time = None
    completion_tokens = None

    # We count "token events" approximately by counting non-empty deltas;
    # this is not exact tokens but correlates with streaming chunks.
    delta_events = 0

    try:
        async with client.stream(
            "POST",
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
            },
            timeout=600,
        ) as r:
            status = r.status_code
            r.raise_for_status()

            async for line in r.aiter_lines():
                if not line:
                    continue
                # SSE format: "data: {...}" or "data: [DONE]"
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    end_time = time.perf_counter()
                    break

                # Parse JSON chunk
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    # Ignore malformed chunk
                    continue

                now = time.perf_counter()

                # Try to detect first content delta (TTFT proxy)
                choices = chunk.get("choices") or []
                if choices:
                    delta = (choices[0].get("delta") or {})
                    content = delta.get("content")
                    if content:
                        delta_events += 1
                        if first_token_time is None:
                            first_token_time = now
                            ttft = first_token_time - t0

                # Some servers include usage in final chunk
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    ct = usage.get("completion_tokens")
                    if isinstance(ct, int):
                        completion_tokens = ct

            if end_time is None:
                end_time = time.perf_counter()

        latency = end_time - t0

        # TPOT (time per token) approximation:
        # If we have completion_tokens >= 2: (end - first) / (tokens - 1)
        # Else if we only have delta events >= 2: (end - first) / (events - 1)
        tpot = None
        if first_token_time is not None:
            denom = None
            if isinstance(completion_tokens, int) and completion_tokens >= 2:
                denom = completion_tokens - 1
            elif delta_events >= 2:
                denom = delta_events - 1
            if denom:
                tpot = (end_time - first_token_time) / denom

        return latency, ttft, tpot, completion_tokens, status, None

    except Exception as e:
        t1 = time.perf_counter()
        status = getattr(getattr(e, "response", None), "status_code", 0) or 0
        return (t1 - t0), None, None, None, status, repr(e)


async def run(
    concurrency: int,
    requests: int,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    limits = httpx.Limits(
        max_connections=max(10, concurrency * 2),
        max_keepalive_connections=max(10, concurrency * 2),
    )
    lats: list[float] = []
    ttfts: List[float] = []
    tpots: List[float] = []
    toks: List[int] = []
    errors: List[str] = []
    status_counts = Counter()

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker():
            async with sem:
                if stream:
                    lat, ttft, tpot, out_tok, status, err = await one_request_stream(
                        client, url, model, max_tokens, temperature, top_p
                    )
                    lats.append(lat)
                    status_counts[status] += 1
                    if ttft is not None:
                        ttfts.append(ttft)
                    if tpot is not None:
                        tpots.append(tpot)
                    if isinstance(out_tok, int):
                        toks.append(out_tok)
                    if err:
                        errors.append(err)
                else:
                    lat, out_tok, status, err = await one_request_nonstream(
                        client, url, model, max_tokens, temperature, top_p
                    )
                    lats.append(lat)
                    status_counts[status] += 1
                    if isinstance(out_tok, int):
                        toks.append(out_tok)
                    if err:
                        errors.append(err)

        # Real wall-clock measurement (rigorous)
        wall_t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(requests)]
        await asyncio.gather(*tasks, return_exceptions=False)
        wall_t1 = time.perf_counter()

    wall = wall_t1 - wall_t0
    ttft_sorted = sorted(ttfts)
    tpot_sorted = sorted(tpots)
    
    total_tokens = sum(toks)
    tps = tokens_per_second(total_tokens, wall)
    rps = (requests / wall) if wall > 0 else None
    err_rate = (len(errors) / requests) if requests > 0 else None
    lat_sum = summarize_latencies(lats)
    report = {
        "mode": "stream" if stream else "nonstream",
        "base_url": base_url,
        "endpoint": "/v1/chat/completions",
        "model": model,
        "concurrency": concurrency,
        "requests": requests,
        "wall_time_s": wall,
        "requests_per_second": rps,
        "error_count": len(errors),
        "error_rate": err_rate,
        "http_status_counts": dict(status_counts),
        "latency": {
            "mean_s": lat_sum.mean_s,
            "p50_s": lat_sum.p50_s,
            "p95_s": lat_sum.p95_s,
            "p99_s": lat_sum.p99_s,
            "min_s": lat_sum.min_s,
            "max_s": lat_sum.max_s,
        },
        "tokens": {
            "completion_tokens_total": total_tokens if total_tokens > 0 else None,
            "tokens_per_second": tps,
            "completion_tokens_available_requests": len(toks),
        },
    }

    if stream:
        report["stream_metrics"] = {
            "ttft": {
                "mean_s": statistics.mean(ttfts) if ttfts else None,
                "p50_s": percentile(ttft_sorted, 50),
                "p95_s": percentile(ttft_sorted, 95),
                "p99_s": percentile(ttft_sorted, 99),
                "available_requests": len(ttfts),
            },
            "tpot": {
                "mean_s": statistics.mean(tpots) if tpots else None,
                "p50_s": percentile(tpot_sorted, 50),
                "p95_s": percentile(tpot_sorted, 95),
                "p99_s": percentile(tpot_sorted, 99),
                "available_requests": len(tpots),
            },
        }

    # Keep errors but cap to avoid huge output
    if errors:
        report["errors_sample"] = errors[:10]

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--stream", action="store_true", help="Enable streaming and measure TTFT/TPOT")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rep = asyncio.run(
        run(
            args.concurrency,
            args.requests,
            args.base_url,
            args.model,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.stream,
        )
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()