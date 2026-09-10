---
title: "glm-ocr"
created: "2026-09-10 14:44:51"
updated: "2026-09-10 14:52:16"
folder: "模型部署/30服务器模型部署"
---

# glm-ocr 模型部署

`
启动脚本地址：/home/admin/middleware/vllm-GLM-OCR/start.sh
`
`
模型地址：/home/admin/models/GLM-OCR/
`

```
#!/bin/bash

docker rm -f vllm-glm-ocr 2>/dev/null || true

docker run -d \
  --name vllm-glm-ocr \
  --restart unless-stopped \
  --gpus '"device=2"' \
  -p 35663:8080 \
  -v /home/admin/models/GLM-OCR:/models/GLM-OCR \
  --entrypoint /bin/bash \
  vllm/vllm-openai:nightly \
  -c '
    exec vllm serve /models/GLM-OCR \
      --port 8080 \
      --served-model-name glm-ocr \
      --speculative-config '"'"'{"method": "mtp", "num_speculative_tokens": 3}'"'"' \
      --gpu-memory-utilization 0.25 \
      --max-model-len 4086 \
      --dtype half \
      --trust-remote-code
  '

```