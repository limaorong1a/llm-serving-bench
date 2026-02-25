cat > run_profile_and_load.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /root/storage-public/llm-serving-bench
mkdir -p traces logs

LOG_FILE="logs/vllm_$(date +%F_%H%M%S).log"
TRACE_PREFIX="traces/vllm_prefill_heavy"

# 1) 在 nsys 下启动 vLLM（后台）
nsys profile \
  -o "$TRACE_PREFIX" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  vllm serve "$HOME/storage-public/limaorong/model/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen2.5-7b \
    --gpu-memory-utilization 0.90 \
  >"$LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "vLLM started under nsys, pid=$VLLM_PID"
# 2) 等服务起来（简单等 3 秒；更严谨可以 curl /v1/models）
sleep 3

# 3) 跑压测（你现在这条）
python -m loadgen.min_loadgen --prompt-repeat 50 --max-tokens 16 --requests 64 --concurrency 4

# 4) 结束 vLLM，让 nsys finalize 输出文件
echo "Stopping vLLM (pid=$VLLM_PID) to finalize nsys trace..."
kill -INT "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" || true

echo "Done. Traces directory:"
ls -lah traces | head -n 50
SH

chmod +x run_profile_and_load.sh
./run_profile_and_load.sh