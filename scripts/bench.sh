cat > scripts/bench.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${SERVED_MODEL_NAME:-qwen2.5-7b}"

mkdir -p results

# 3 组标准负载（你也可以改成你想要的）
python loadgen/min_loadgen.py --base-url "$BASE_URL" --model "$MODEL" \
  --concurrency 1 --requests 20 --out results/c1.json

python loadgen/min_loadgen.py --base-url "$BASE_URL" --model "$MODEL" \
  --concurrency 4 --requests 40 --out results/c4.json

python loadgen/min_loadgen.py --base-url "$BASE_URL" --model "$MODEL" \
  --concurrency 16 --requests 80 --out results/c16.json

echo "Saved: results/c1.json results/c4.json results/c16.json"
EOF

chmod +x scripts/bench.sh