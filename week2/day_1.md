#trace-prefill_heavy
压测结果：
{
  "mode": "nonstream",
  "base_url": "http://127.0.0.1:8000",
  "endpoint": "/v1/chat/completions",
  "model": "qwen2.5-7b",
  "concurrency": 16,
  "requests": 512,
  "wall_time_s": 12.293269098037854,
  "requests_per_second": 41.64880764561813,
  "error_count": 0,
  "error_rate": 0.0,
  "http_status_counts": {
    "200": 512
  },
  "latency": {
    "mean_s": 0.379341736259903,
    "p50_s": 0.3585015576099977,
    "p95_s": 0.4382950782543048,
    "p99_s": 0.9500414061662741,
    "min_s": 0.27897925791330636,
    "max_s": 0.951869711978361
  },
  "tokens": {
    "completion_tokens_total": 8192,
    "tokens_per_second": 666.38092232989,
    "completion_tokens_available_requests": 512
  }
}
prefill-heavy 阶段总体呈现‘大 kernel 主导’，GPU 主要在跑 BF16 GEMM（Cutlass）和 FlashAttention forward；因此它更接近算力/带宽型瓶颈，理论上更容易吃满 GPU。当前 trace 的 CUDA API summary 受到 cudagraph capture/warmup 影响，后续需在 steady-state 窗口内复测确认同步与 H2D 是否仍显著。

#trace-decode_heavy
压测结果：
{
  "mode": "nonstream",
  "base_url": "http://127.0.0.1:8000",
  "endpoint": "/v1/chat/completions",
  "model": "qwen2.5-7b",
  "concurrency": 8,
  "requests": 128,
  "wall_time_s": 26.973620352102444,
  "requests_per_second": 4.745377088026788,
  "error_count": 0,
  "error_rate": 0.0,
  "http_status_counts": {
    "200": 128
  },
  "latency": {
    "mean_s": 1.6651919576088403,
    "p50_s": 1.6312936380272731,
    "p95_s": 2.1574219137197357,
    "p99_s": 2.8891387366270656,
    "min_s": 1.1834447081200778,
    "max_s": 3.3554086359217763
  },
  "tokens": {
    "completion_tokens_total": 20087,
    "tokens_per_second": 744.6905434937039,
    "completion_tokens_available_requests": 128
  }
}
decode 慢的结构性原因是：每 token 都要执行 attention + KV 访问 + FFN(GEMM) + 采样，并且由于 batch 动态变化导致计算更碎、kernel 更小更频繁，launch/sync 开销被放大。
decode 阶段 active 序列会因为 EOS/插单而持续变化，且各序列 KV 长度不一致，导致每一步能凑出的有效 batch 规模和形状不稳定。结果是：FFN/attention/采样被拆成大量短小 kernel（micro-kernels）高频执行，kernel launch 与同步/等待的固定开销占比上升，从而表现为计算更碎、调度更重。
##decode结构性慢的原因：
-attention + KV 访问：每 token 都要做一次（而且 KV 越长，访问越重）
-FFN/GEMM：仍是时间大头（ GPU kernel 时间 83.6% 都是 cutlass GEMM）
-采样：NVTX 里 RadixSort 刷屏（top-k/top-p）
-同步开销：cudaEventSynchronize 占比极高（说明 pipeline 有很多“必须等一下”的点）


KV cache 在 prefill 与 decode 是同一份缓存；差异在访问方式：prefill 一次写入一整段 prompt 的 KV、计算形状更容易按相近 prompt 长度凑批而规整；decode 每步只追加 1 token，但每步都要读全历史 KV，且活跃序列长度因插入/早停持续分化，导致 batch 的 KV 长度天然不一致、更碎。