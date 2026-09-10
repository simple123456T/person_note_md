---
title: "minicpm-v4.6"
created: "2026-09-10 14:53:56"
updated: "2026-09-10 14:55:08"
folder: "模型部署/30服务器模型部署"
---

# minicpm-v4.6 模型部署

`
启动脚本地址：/home/admin/middleware/vllm-MiniCPM-V-4.6/start.sh
`
`
模型地址：/home/admin/models/MiniCPM-V-4.6/
`

```
docker rm -f vllm-minicpm-v46 2>/dev/null || true

docker run -d \
  --gpus '"device=2"' \
  -p 36523:8000 \
  -v /home/admin/models/MiniCPM-V-4.6:/models \
  --name vllm-minicpm-v46 \
  vllm/vllm-openai:v0.24.0 \
  --model /models \
  --served-model-name MiniCPM-V-4.6 \
  --trust-remote-code \
  --max-model-len 2048 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 1}'

```