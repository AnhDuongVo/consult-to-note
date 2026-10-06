#!/usr/bin/env bash
# NVIDIA Dynamo, disaggregated serving on two GPUs: GPU 0 decodes, GPU 1 runs prefill, KV cache moves
# between them with NIXL. Follows examples/backends/vllm/launch/disagg.sh in the Dynamo repo.
#
# Why try it: prefill (reading the long transcript) is compute-bound and decode (writing the note) is
# memory-bandwidth-bound. Separating them lets long prompts stop slowing down token generation for
# everyone else, which shows up as a lower and steadier TTFT / ITL at higher concurrency.
#
# Note: upstream's quick-start adds --enforce-eager (no CUDA graphs). It is left out here so the numbers are
# comparable with aggregated.sh. If workers do not find each other, see the Dynamo docs on discovery.
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
PORT="${PORT:-8001}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT

python3 -m dynamo.frontend --http-port "$PORT" &

DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model "$MODEL"  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' $EXTRA_ARGS &

DYN_SYSTEM_PORT=8082 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
  --model "$MODEL"  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' \
  $EXTRA_ARGS &

until curl -sf "localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
echo "Disaggregated Dynamo serving $MODEL on :$PORT. Ctrl+C to stop."
wait
