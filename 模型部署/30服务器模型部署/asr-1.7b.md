---
title: "asr-1.7b"
created: "2026-09-10 14:55:37"
updated: "2026-09-10 14:57:41"
folder: "模型部署/30服务器模型部署"
---

# asr-1.7b 模型部署

`
启动脚本地址：/home/admin/middleware/vllm-Qwen3-ASR-1.7B/start.sh
`
`
模型地址：/home/admin/models/Qwen3-ASR-1.7B/
`

``` start.sh
#!/bin/bash

# 1. 删除可能存在的同名容器（忽略不存在时的错误）
docker rm -f vllm-qwen3-asr-1.7b 2>/dev/null || true

# 2. 启动服务（使用 docker-compose）
docker compose up -d

# 3. （可选）查看容器状态
docker ps --filter "name=vllm-qwen3-asr-1.7b"
```

``` docker-compose.yaml
services:
  qwen3-asr:
    image: qwenllm/qwen3-asr:latest
    container_name: vllm-qwen3-asr-1.7b
    restart: unless-stopped

    ports:
      - "17860:7860"

    volumes:
      - /home/admin/models/Qwen3-ASR-1.7B:/models/Qwen3-ASR-1.7B

    environment:
      VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: "100"

    command: >
      qwen-asr-serve /models/Qwen3-ASR-1.7B
      --served-model-name Qwen3-ASR-1.7B
      --gpu-memory-utilization 0.45
      --max-model-len 32768
      --dtype half
      --host 0.0.0.0
      --port 7860

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]

```