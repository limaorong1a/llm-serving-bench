#!/usr/bin/env bash
set -euo pipefail

cd /root/storage-public/llm-serving-bench
mkdir -p traces logs

# hard cleanup to avoid hitting old server
pkill -f "vllm serve .*--port 8000" || true
pkill -f "VLLM::EngineCore" || true
pkill -f "nsys profile .*vllm_decode_heavy" || true
sleep 1
ss -lntp | grep ':8000' && echo "ERROR: 8000 still in use" || echo "8000 free"

LOG_FILE="logs/vllm_$(date +%F_%H%M%S).log"
TRACE_PREFIX="traces/vllm_decode_heavy"

# 1) 在 nsys 下启动 vLLM（后台）
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  -o "$TRACE_PREFIX" \
  --force-overwrite=true \
  --trace=cuda-sw,nvtx,osrt \
  --sample=none \
  --cuda-trace-scope=process-tree \
  --wait=all \
  --cuda-flush-interval=100\
  vllm serve "$HOME/storage-public/limaorong/model/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen2.5-7b \
    --gpu-memory-utilization 0.90 \
  >"$LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "vLLM started under nsys, pid=$VLLM_PID"
# 2) 等服务起来（简单等 3 秒；更严谨可以 curl /v1/models）
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models || true)
  if [[ "$code" == "200" ]]; then
    echo "vLLM ready"
    break
  fi
  sleep 1
done


# 3) 跑压测（你现在这条）
python -m loadgen.min_loadgen --prompt-repeat 1 --max-tokens 512 --requests 128 --concurrency 8

# 4) 结束 vLLM，让 nsys finalize 输出文件
echo "Stopping vLLM (nsys pid=$VLLM_PID) to finalize nsys trace..."
PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
echo "Sending SIGINT to process group PGID=$PGID"
kill -INT -- -"${PGID}" 2>/dev/null || true
wait "$VLLM_PID" || true

echo "Done. Traces directory:"
ls -lah traces | head -n 50
