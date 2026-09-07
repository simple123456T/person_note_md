---
title: "qwen3-asr-1.7b"
created: "2026-09-07 09:56:01"
updated: "2026-09-07 09:57:18"
folder: "模型部署/198服务器模型部署"
---

# 198服务器 Qwen3-ASR-1.7B部署 脚本

1.7B 参数量模型占用 10G 显存

```docker-compose.yaml
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
      - NVIDIA_VISIBLE_DEVICES="7"
      - VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=100
    command: >
      qwen-asr-serve /models/Qwen3-ASR-1.7B
      --gpu-memory-utilization 0.45
      --dtype half
      --host 0.0.0.0 --port 7860
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```