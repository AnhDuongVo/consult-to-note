#!/usr/bin/env bash
# NVIDIA Dynamo, aggregated serving on one GPU (vLLM backend), OpenAI-compatible API on :8001.
#
#   pip install uv && uv pip install --prerelease=allow "ai-dynamo[vllm]"
#   # or use the container: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>
#   bash deploy/dynamo/aggregated.sh
#
# Smoke test with a tiny model first:  MODEL=Qwen/Qwen3-0.6B bash deploy/dynamo/aggregated.sh
# Production-grade Kubernetes recipes (e.g. Nemotron 3.5 Lightning in NVFP4 with speculative decoding):
#   https://github.com/ai-dynamo/dynamo  -> docs "Recipes"
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
PORT="${PORT:-8001}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. "--max-model-len 32768 --trust-remote-code"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT

# File-based discovery: no etcd / NATS needed for a single machine.
python3 -m dynamo.frontend --http-port "$PORT" --discovery-backend file &
python3 -m dynamo.vllm --model "$MODEL" --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}' $EXTRA_ARGS &

echo "Waiting for $MODEL on :$PORT ..."
until curl -sf "localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
curl -s "localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say ready.\"}],\"max_tokens\":10}"
echo
echo "Dynamo is serving. Ctrl+C to stop."
wait
