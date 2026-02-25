LOG_DIR="$HOME/storage-public/limaorong/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_qwen2.5-7b_$(date +%F_%H%M%S).log"

nohup vllm serve "$HOME/storage-public/limaorong/model/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2.5-7b \
  --gpu-memory-utilization 0.90 \
  >"$LOG_FILE" 2>&1 < /dev/null &
disown

echo "Started. Log: $LOG_FILE"