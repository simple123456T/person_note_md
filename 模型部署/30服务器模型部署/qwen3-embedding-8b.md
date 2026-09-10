---
title: "qwen3-embedding-8b"
created: "2026-09-07 10:28:34"
updated: "2026-09-10 14:41:01"
folder: "模型部署/30服务器模型部署"
---

# qwen3-embedding-8b 模型部署
`
启动脚本地址：/home/admin/middleware/vllm-Qwen3-Embedding-8B/start.sh
`
`
模型地址：/home/admin/models/Qwen--Qwen3-Embedding-8B
`
```
docker rm -f vllm-Qwen3-Embedding-8B 2>/dev/null || true

docker run -d \
  --name vllm-Qwen3-Embedding-8B \
  --gpus '"device=1"' \
  -p 54214:8000 \
  -v /home/admin/models/Qwen--Qwen3-Embedding-8B/Qwen/Qwen3-Embedding-8B:/models/Qwen3-Embedding-8B:ro \
  --restart=unless-stopped \
  vllm/vllm-openai:v0.17.0 \
  /models/Qwen3-Embedding-8B \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.95 \
  --dtype float16 \
  --served-model-name ph-Qwen-Embedding-8B
```

# 测试脚本

```
curl --location --request POST 'http://192.168.2.30:54214/v1/embeddings' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model_name": "ph-Qwen-Embedding-8B",
  "input": [
    "这是第一个测试文本",
    "这是第二个测试文本",
    "这是第三个测试文本"
  ]
}'
```